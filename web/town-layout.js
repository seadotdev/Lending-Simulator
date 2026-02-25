/**
 * Pure data layout algorithm for the town.
 * All positions are in world units; road tile = 1x1 unit.
 */

const TILE_SIZE = 2;        // world units per road tile
const BUILDING_OFFSET = 3;  // distance from road center to building center
const PROP_OFFSET = 1.6;    // distance from road center to props

const DEFAULT_BUILDING_FILES = [
  'building_A.gltf', 'building_B.gltf', 'building_C.gltf', 'building_D.gltf',
  'building_E.gltf', 'building_F.gltf', 'building_G.gltf', 'building_H.gltf',
];

const PROP_CYCLE = ['streetlight', 'bench', 'bush', 'firehydrant', 'bush', 'bench'];

export function generateLayout(lenderCount, buildingFiles) {
  const BUILDING_FILES = buildingFiles || DEFAULT_BUILDING_FILES;
  const rows = Math.ceil(lenderCount / 2);
  const roads = [];
  const buildings = [];
  const props = [];

  // Road tiles along Z axis
  for (let r = 0; r < rows; r++) {
    roads.push({
      x: 0,
      z: r * TILE_SIZE,
      rotation: 0,
      modelFile: 'road_straight.gltf',
    });
  }

  // Buildings on alternating sides
  for (let i = 0; i < lenderCount; i++) {
    const row = Math.floor(i / 2);
    const side = i % 2 === 0 ? -1 : 1;  // left or right of road
    const tier = Math.min(Math.floor(i * BUILDING_FILES.length / lenderCount), BUILDING_FILES.length - 1);

    buildings.push({
      id: i,
      x: side * BUILDING_OFFSET,
      z: row * TILE_SIZE,
      rotation: side === -1 ? Math.PI / 2 : -Math.PI / 2,
      modelFile: BUILDING_FILES[tier],
    });
  }

  // Props along road edges
  for (let r = 0; r < rows; r++) {
    const propName = PROP_CYCLE[r % PROP_CYCLE.length];
    // Left side prop
    props.push({
      x: -PROP_OFFSET,
      z: r * TILE_SIZE + TILE_SIZE * 0.4,
      rotation: 0,
      propType: propName,
    });
    // Right side prop
    props.push({
      x: PROP_OFFSET,
      z: r * TILE_SIZE + TILE_SIZE * 0.4,
      rotation: Math.PI,
      propType: propName === 'streetlight' ? 'streetlight' : 'bush',
    });
  }

  return {
    buildings,
    roads,
    props,
    tileSize: TILE_SIZE,
    totalLength: rows * TILE_SIZE,
    buildingOffset: BUILDING_OFFSET,
  };
}
