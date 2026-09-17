# Project website for 21cm-sbi-vmim

Static site, no build step. Put these files in the `docs/` folder of the repository,
then choose Settings > Pages > Deploy from branch > `main` / `docs`.

## Posterior explorer

`explorer.html` + `explorer.js` read the files written by `export_web.py`:

    python export_web.py --list sbc_lists/<arms>.list --out docs/data

Preview locally (fetch() does not work from file://):

    cd docs && python -m http.server 8000
    # open http://localhost:8000/explorer.html

Shareable links: explorer.html#sim=1025&arms=cnn_vmim_n1,cnn_mse_n1&bins=30
