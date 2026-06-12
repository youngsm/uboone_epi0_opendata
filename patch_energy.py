"""Patch a make_pilarnet.py output produced BEFORE the energy-convention
fix (when 'true_energy' stored TOTAL energy) up to the current schema,
without re-running the solver.

Recoverable exactly from total energy + label:
    true_momentum     = sqrt(E_tot^2 - m^2)
    true_ke           = E_tot - m
    true_energy_total = E_tot (copied)
    true_energy       -> overwritten with momentum (current default alias)
NOT recoverable: visible_energy (needs edep_table); re-run the converter
only if you decide the paper bins in deposited energy.

Usage: python patch_energy.py pilarnet_epi0.h5
"""
import sys
import numpy as np
import h5py

M = {0: 0.000511, 1: 0.13498}   # e, pi0 [GeV]

path = sys.argv[1]
with h5py.File(path, 'r+') as f:
    if 'true_momentum' in f:
        sys.exit('already current schema; nothing to do')
    etot = f['true_energy'][:].astype(np.float64)
    mass = np.array([M[int(l)] for l in f['label'][:]])
    p = np.sqrt(np.maximum(etot ** 2 - mass ** 2, 0.0))
    f.create_dataset('true_energy_total', data=etot.astype(np.float32))
    f.create_dataset('true_momentum', data=p.astype(np.float32))
    f.create_dataset('true_ke', data=(etot - mass).astype(np.float32))
    del f['true_energy']
    f.create_dataset('true_energy', data=p.astype(np.float32))
    f.attrs['energy_convention'] = (
        'patched: true_energy == true_momentum (GeV); '
        'visible_energy absent (requires converter re-run)')
print('patched', path, f'({len(etot)} events)')
