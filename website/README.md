# Project page

A static page with no build step: `index.html` plus `static/`. It is published to GitHub Pages
by `.github/workflows/pages.yml` on every push to `main` that touches this folder (enable Pages in
the repository settings with **GitHub Actions** as the source).

Open `index.html` directly in a browser, or serve the folder:

```bash
python -m http.server --directory website 8000    # then open http://localhost:8000
```

`static/` holds the change gallery, the frame sequences of the arm viewer (re-rendered from stored actions of held-out FLOWER cases) and two result figures.

`static/fl-page.css` and `static/fl-page.js` are the interactive layer shared by the Forall Labs
project pages (OpenCaP, AutoPolish, CALVIN-MAX): the hero with a visual beside the title, the sticky
section bar, the compare slider, figure zoom and the hover states. The same two files ship in each
repository, so edit them in one and copy them to the others.
