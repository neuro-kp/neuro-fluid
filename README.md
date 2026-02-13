# neuro-fluid

Experiments in data provenance and neuro-imaging with Fluid.

## Dev setup

### Software required
- Node.js >=18.0.0
- yarn >= 1.22

### Dev tools
- There is no language plugin for Fluid yet, but any Python syntax highlighter should do a reasonable job

### Initial configuration
- `yarn install` to (re-)install Node dependencies
- `yarn install-website article` to (re-)install example article website from `@exploreable-viz/fluid`
- Add `dist/` and `website/` folders to `.gitignore`

### Bundling and serving website
For arbitrary `$WEBSITE_NAME` (e.g. `article`):

- `yarn bundle-website $WEBSITE_NAME` to bundle website to `dist/$WEBSITE_NAME`
- `npx http-serve dist/$WEBSITE_NAME -c-1` to serve website at localhost
