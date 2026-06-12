// pybind11 binding around the ACTUAL larreco SpacePointSolver charge solver.
//
// Compiled sources: larreco/SpacePointSolver/{Solver.cxx, QuadExpr.cxx} from
// github.com/LArSoft/larreco @ 6c4c0fd918a5577feeaa3865e5229de7b36f075c
// (byte-identical at the OpenSamples-era tag LARSOFT_SUITE_v08_05_00_17),
// compiled UNMODIFIED. This file only reproduces the orchestration that
// lives in SpacePointSolver_module.cc (which cannot compile outside the art
// framework), with line references:
//   * system construction order   <- BuildSystem,            #L273-L363
//     (note: CollectionWireHit ctor performs the equal q/N split BEFORE
//      AddNeighbours runs at #L362, so the initial split does NOT seed
//      fNeiPotential — preserved here by construction order)
//   * neighbour assignment        <- AddNeighbours,          #L175-L252
//     (kCritDist=5 #L177, coupling=exp(-d/2) #L248, d==0 skip #L237;
//      O(N^2) scan here instead of the grid buckets — same pair set)
//   * Minimize loop, verbatim     <- #L436-L454
//     (prevMetric before loop, stop on increase, |dM| < 1e-3|M|)
//   * orphan list is empty: orphans require bad collection channels,
//     not modeled from the OpenSamples HDF5 (no channel-status DB).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include "Solver.h"

namespace py = pybind11;

static void Minimize(const std::vector<CollectionWireHit*>& cwires,
                     const std::vector<SpaceCharge*>& orphanSCs,
                     double alpha, int maxiterations)
{
  // verbatim SpacePointSolver_module.cc#L436-L454, minus the couts
  double prevMetric = Metric(cwires, alpha);
  for(int i = 0; i < maxiterations; ++i){
    Iterate(cwires, orphanSCs, alpha);
    const double metric = Metric(cwires, alpha);
    if(metric > prevMetric) return;
    if(fabs(metric-prevMetric) < 1e-3*fabs(prevMetric)) return;
    prevMetric = metric;
  }
}

// Inputs (system already triplet-found; indices are dense 0..M-1 / 0..K-1):
//   xyz   (N,3) SpaceCharge positions
//   iw1   (N,)  index of U-plane InductionWireHit per SpaceCharge
//   iw2   (N,)  index of V-plane InductionWireHit per SpaceCharge
//   cw    (N,)  index of CollectionWireHit per SpaceCharge
//   iwq   (M,)  induction hit charges (Integral)
//   cwq   (K,)  collection hit charges (Integral)
// Returns (N,) solved charges (SpaceCharge::fPred after both phases).
py::array_t<double> solve_system(
    py::array_t<double, py::array::c_style | py::array::forcecast> xyz,
    py::array_t<long>   iw1, py::array_t<long> iw2, py::array_t<long> cw,
    py::array_t<double> iwq, py::array_t<double> cwq,
    double alpha, int max_iter_noreg, int max_iter_reg, double crit_dist)
{
  const auto X = xyz.unchecked<2>();
  const auto I1 = iw1.unchecked<1>(), I2 = iw2.unchecked<1>(),
             C = cw.unchecked<1>();
  const auto QI = iwq.unchecked<1>(), QC = cwq.unchecked<1>();
  const long N = X.shape(0), M = QI.shape(0), K = QC.shape(0);

  // BuildSystem #L116-L122: induction wires (chan = index; unused)
  std::vector<InductionWireHit*> iwires;
  iwires.reserve(M);
  for(long m = 0; m < M; ++m)
    iwires.push_back(new InductionWireHit(int(m), QI(m)));

  // #L129-L137: SpaceCharges, cwire pointer set later
  std::vector<SpaceCharge*> scs;
  scs.reserve(N);
  std::vector<std::vector<SpaceCharge*>> byCW(K);
  for(long n = 0; n < N; ++n){
    SpaceCharge* sc = new SpaceCharge(X(n,0), X(n,1), X(n,2),
                                      0, iwires[I1(n)], iwires[I2(n)]);
    scs.push_back(sc);
    byCW[C(n)].push_back(sc);
  }

  // #L153-L173: CollectionWireHit ctor performs the equal split
  // (Solver.cxx#L47); neighbours are NOT yet assigned, as in the module.
  std::vector<CollectionWireHit*> cwires;
  cwires.reserve(K);
  for(long k = 0; k < K; ++k){
    if(byCW[k].empty()) continue;
    CollectionWireHit* cwire = new CollectionWireHit(int(k), QC(k), byCW[k]);
    for(SpaceCharge* sc: byCW[k]) sc->fCWire = cwire;
    cwires.push_back(cwire);
  }

  // AddNeighbours #L175-L252 (only when alpha != 0, per #L362 incNei)
  if(alpha != 0){
    for(long a = 0; a < N; ++a){
      for(long b = 0; b < N; ++b){
        if(a == b) continue;
        const double dx = X(a,0)-X(b,0), dy = X(a,1)-X(b,1),
                     dz = X(a,2)-X(b,2);
        const double d2 = dx*dx + dy*dy + dz*dz;
        if(d2 > crit_dist*crit_dist || d2 == 0) continue;
        scs[a]->fNeighbours.emplace_back(scs[b], std::exp(-std::sqrt(d2)/2));
      }
    }
  }

  // produce() #L362-L370: noreg phase then reg phase, no orphans
  const std::vector<SpaceCharge*> orphans;
  Minimize(cwires, orphans, 0, max_iter_noreg);
  Minimize(cwires, orphans, alpha, max_iter_reg);

  py::array_t<double> out(N);
  auto O = out.mutable_unchecked<1>();
  for(long n = 0; n < N; ++n) O(n) = scs[n]->fPred;

  for(CollectionWireHit* c: cwires) delete c;   // dtor deletes its SCs
  for(InductionWireHit* i: iwires) delete i;
  return out;
}

PYBIND11_MODULE(spsolver_cpp, m){
  m.doc() = "Actual larreco SpacePointSolver charge solver (Solver.cxx "
            "compiled unmodified), system construction per "
            "SpacePointSolver_module.cc";
  m.def("solve_system", &solve_system,
        py::arg("xyz"), py::arg("iw1"), py::arg("iw2"), py::arg("cw"),
        py::arg("iwq"), py::arg("cwq"), py::arg("alpha"),
        py::arg("max_iter_noreg"), py::arg("max_iter_reg"),
        py::arg("crit_dist") = 5.0);
}
