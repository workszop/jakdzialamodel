# Fruit illustration provenance

Two existing illustrations were downloaded unchanged for the planned strawberry/blueberry training demo. Both PNGs were visually inspected; each is 194 × 192 pixels, RGBA, and includes its original white card and black border. They are illustrations, not a photographic training dataset.

- Author/repository: [aegeorge42, Neural Networks from Scratch](https://github.com/aegeorge42/aegeorge42.github.io).
- Pinned source commit: `a5f530d4ca61138383e69341c3a4854d50f3f0f6`.
- Repository license: [CC0 1.0 Universal](https://github.com/aegeorge42/aegeorge42.github.io/blob/a5f530d4ca61138383e69341c3a4854d50f3f0f6/LICENSE). The repository README also explicitly describes the project as CC0.
- `strawberry.png`: [original singlestraw.png](https://raw.githubusercontent.com/aegeorge42/aegeorge42.github.io/a5f530d4ca61138383e69341c3a4854d50f3f0f6/images/intro/singlestraw.png).
- `blueberry.png`: [original singleblue.png](https://raw.githubusercontent.com/aegeorge42/aegeorge42.github.io/a5f530d4ca61138383e69341c3a4854d50f3f0f6/images/intro/singleblue.png).

Intended use: sample and output-class illustrations beside an explicitly feature-based classifier. The model receives two synthetic, supplied numerical features, length and roundness. It does not read pixels, extract these features from images, or learn image recognition. Repeated use of one class illustration must not imply distinct training photographs.

The source files remain here for traceability. At implementation time embed their bytes as data URLs in the self-contained HTML, with PL/EN alt text; do not add runtime fetches or remote image dependencies. A short source credit is recommended even though CC0 does not require attribution.

Only the reference application's final training sandbox informs the planned interaction: training controls, changing weights, decision regions, and error/progress. Its earlier lesson sequence, Pixi implementation, and full UI are not being copied.
