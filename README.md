# network-metacommunity-restoration
A network metacommunity framework for restoration

This code reproduces Figure 5 of *"A network–metacommunity framework for ecosystem restoration forecasting"*.
The example uses three woodland birds in Snowdonia (Eryri).

| Species | Common name |
|---|---|
| *Cyanistes caeruleus* | Blue tit |
| *Parus major* | Great tit |
| *Periparus ater* | Coal tit |


## Files


`data/animals/network.pkl` contains the habitat network used in the paper. These four files reproduce Figure 5:

```
figure5.py
habitat_network.py
restoration_forecast.py
data/animals/network.pkl
```
Note: GBIF records are updated over time, so a rebuilt network can differ from the Figure 5.

## Run

```bash
pip install numpy scipy pandas matplotlib rasterio scikit-learn scikit-image openpyxl

python fetch_data.py                
python figure5.py                   # forecast and panels (builds the network if data/animals/network.pkl is missing
python figure5.py --rebuild         # rebuild the habitat network from the cached data
python figure5.py --plots-only      # redraw panels from data/animals/trajectories.npz
```

Output figures: 

| File in `output/` | Panel |
|---|---|
| `habitat_corridor_no_caption` | 5A habitat map and H1–H9 corridor |
| `disturbance` | 5B great tit in H9 with and without the corridor |
| `reintroduction` | 5C R0, R1 and R2a |
| `habitat_improvement` | 5D R0, R1 and R2b |

These panels are assembled manually in PowerPoint.

Key results are written to `data/animals/model_summary.json`: the great tit in H9 declines by 60% after the H1–H9 corridor is lost, and R2b requires its K in H9 to rise from 1 to 4.85 (corresponding to habitat recovery/improvement).

## Step 1: habitat network (`build_network`)

1. Filter GBIF records (2000–2025): remove coordinate uncertainty > 1000 m, duplicate coordinates and duplicate raster cells.
2. Average the environmental rasters to a four-fold coarser grid (about 100 m).
3. Fit MaxEnt to each species as a penalised presence-vs-background logistic regression (linear, quadratic and hinge features; 10,000 background points).
4. Define habitat patches as cells with high suitability (top 15%) and high occurrence density, then merge patches across species.
5. Set K of a species in a patch as its summed suitability over the patch; K = 1 outside the species' habitat cores.
6. Build resistance from slope and the inverse of the best suitability across species; least-cost paths between patches give the distances x.
7. Calculate dispersal for each species as D = 0.1 · exp(−x / α), where α scales with the AVONET hand-wing index.

## Step 2: forecasting model (`forecast`)

For species *i* in habitat *p*:

    dN_ip/dt = N_ip (r_i + Σ_j A_ij N_jp − N_ip / K_pi) + Σ_q D_i,pq N_iq − N_ip Σ_q D_i,qp

- r_i decreases with AVONET body mass; A_ij scales with the body-mass ratio and is the same in every habitat.
- Environmental noise is an AR(1) perturbation of K; demographic noise is lognormal. Seeds are fixed, so all scenarios share the same noise.
- Scaling constants are illustrative; time and abundance are in model units.

Scenario: the H1–H9 corridor is lost at t = 5. Restoration starts at t = 20 with one of:

- R0: no restoration.
- R1: rebuild the corridor.
- R2a: reintroduce great tits in H9.
- R2b: raise K of the great tit in H9 and reintroduce. K is found by bisection so that R2b sustains the same mean abundance as R1 over t = 25–40.

## Data sources

- Occurrences: GBIF occurrence API; query and filters in `data/animals/provenance.json`.
- Traits: AVONET (Tobias et al. 2022), https://doi.org/10.6084/m9.figshare.16586228.v7 (CC BY 4.0).
- Environmental raster: WorldClim, SoilGrid

## Reference

Tobias, J.A. et al. (2022) AVONET: morphological, ecological and geographical data for all birds. *Ecology Letters*, 25, 581–597. https://doi.org/10.1111/ele.13898


