# Building & publishing the documentation site

This project's docs are a **Jupyter Book 2** site (built on the MyST engine),
configured by `myst.yml` at the repo root.

## Preview locally

```bash
pip install "jupyter-book>=2"
jupyter book start          # live-reloading preview at http://localhost:3000
```

## Build static HTML

```bash
jupyter book build --html   # output in _build/html/
```

## Publish to GitHub Pages (automated)

`.github/workflows/deploy-book.yml` rebuilds and publishes the site on every
push to `main`. **One-time setup:** in the repo on GitHub, go to
**Settings → Pages → Build and deployment**, set **Source = "GitHub Actions"**.
After the next push, the site is live at
`https://Dilru1.github.io/21cm-sbi-vmim/`.

## Notes

- The table of contents lives under `project.toc` in `myst.yml`. Add a page by
  adding its file there.
- Notebooks are rendered from their **saved outputs** (not executed at build),
  so the site builds on a bare runner with no cluster data. Run report
  notebooks on the cluster and commit them with outputs to publish their figures.
