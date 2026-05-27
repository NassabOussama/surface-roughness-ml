# Sample images

A small, representative subset of the surface-roughness dataset, provided so
anyone can try the model (e.g. the Streamlit demo) **without needing the full
private dataset**.

Six microscopy images — two per roughness class — chosen as clear, typical
examples of each class (not borderline cases near the Smooth/Medium/Rough cut
points at Ra = 0.653 / 0.960 µm).

## Using them with the model

The model needs **two inputs**: the image **and** the grit value used to machine
that surface. When testing a sample below, enter the grit value from the table.
Each filename also encodes the sample's true mean roughness (`ra_<value>`) so you
can check the prediction against the ground truth.

## Provenance

These are downsampled copies of the original microscopy scans. The originals are
4000×3000 RGBA PNGs (~20 MB each); the copies here are resized to a 1024 px long
edge and converted to RGB (~1 MB each) to keep the repository lightweight. This
has no effect on inference — the model resizes every input to 224×224 internally.

Each copy is **angle 1 of 4** of its sample (every physical sample was
photographed from four angles in the source dataset).

| File | Class | Ra_Moyenne (µm) | Grit | Original sample ID |
|------|-------|-----------------|------|--------------------|
| `smooth_ra_0.45.png` | Smooth | 0.446 | 400 | 40A |
| `smooth_ra_0.52.png` | Smooth | 0.522 | 320 | 33A |
| `medium_ra_0.78.png` | Medium | 0.781 | 180 | 23B |
| `medium_ra_0.84.png` | Medium | 0.842 | 150 | 20A |
| `rough_ra_1.12.png`  | Rough  | 1.120 | 100 | 10A |
| `rough_ra_1.55.png`  | Rough  | 1.548 | 60  | 1B  |

*Original sample ID is `<Numero><Face>` from `Labelisation_CSV.csv`; the source
file for each was `<ID>_1.png` in the full image set.*
