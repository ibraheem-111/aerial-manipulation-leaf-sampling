"""Repository-relative folders, so the code runs wherever the repo is cloned.

Layout: <repo>/src/uam_quad/uam_quad/paths.py  ->  REPO = <repo>
Outputs go to <repo>/runs, generated ACADOS solvers to <repo>/build (both git-ignored).
"""
import os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # <repo>/src/uam_quad
REPO = os.path.dirname(os.path.dirname(PKG))                          # <repo>
RUNS = os.path.join(REPO, "runs", "uam_quad")
BUILD = os.path.join(REPO, "build")
MODELS = os.path.join(PKG, "models")
