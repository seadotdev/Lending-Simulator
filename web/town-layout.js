/**
 * Pure data layout algorithm for the town.
 * L-shaped town: residential zone along -Z axis, financial district along +X axis.
 * Junction at origin (0,0,0) connects the two roads.
 */

const TILE_SIZE = 2;        // world units per road tile
const BUILDING_OFFSET = 3;  // distance from road center to building center
const PROP_OFFSET = 1.6;    // distance from road center to props

const DEFAULT_BUILDING_FILES = [
  'building_A.gltf', 'building_B.gltf', 'building_C.gltf', 'building_D.gltf',
  'building_E.gltf', 'building_F.gltf', 'building_G.gltf', 'building_H.gltf',
];

const TOWN_BUILDING_FILES = [
  'building_A.gltf', 'building_B.gltf', 'building_C.gltf', 'building_D.gltf',
  'building_E.gltf', 'building_F.gltf', 'building_G.gltf', 'building_H.gltf',
];

const PROP_CYCLE = ['streetlight', 'bench', 'bush', 'firehydrant', 'bush', 'bench'];

const RESIDENTIAL_ROWS = 4;  // rows of houses in residential zone

export function generateLayout(lenderCount, buildingFiles) {
  const BUILDING_FILES = buildingFiles || DEFAULT_BUILDING_FILES;
  const bankRows = Math.ceil(lenderCount / 2);
  const roads = [];
  const buildings = [];   // bank buildings (lender-indexed)
  const townBuildings = []; // decorative residential buildings
  const props = [];
  const vehicles = [];

  // ---- Junction tile at origin ----
  roads.push({
    x: 0, z: 0, rotation: 0,
    modelFile: 'road_tsplit.gltf',
  });

  // ---- Financial district: road tiles along +X axis ----
  for (let c = 1; c <= bankRows; c++) {
    roads.push({
      x: c * TILE_SIZE, z: 0, rotation: Math.PI / 2,
      modelFile: 'road_straight.gltf',
    });
  }

  // ---- Bank buildings along +X road, alternating north/south ----
  for (let i = 0; i < lenderCount; i++) {
    const col = Math.floor(i / 2) + 1; // start at column 1 (column 0 is junction)
    const side = i % 2 === 0 ? -1 : 1; // north (-Z side) or south (+Z side)
    const tier = Math.min(
      Math.floor(i * BUILDING_FILES.length / lenderCount),
      BUILDING_FILES.length - 1
    );

    buildings.push({
      id: i,
      x: col * TILE_SIZE,
      z: side * BUILDING_OFFSET,
      rotation: side === -1 ? 0 : Math.PI, // face toward the road
      modelFile: BUILDING_FILES[tier],
    });
  }

  // ---- Residential zone: road tiles along -Z axis ----
  for (let r = 1; r <= RESIDENTIAL_ROWS; r++) {
    roads.push({
      x: 0, z: -r * TILE_SIZE, rotation: 0,
      modelFile: 'road_straight.gltf',
    });
  }

  // ---- Decorative town buildings along residential road ----
  for (let r = 1; r <= RESIDENTIAL_ROWS; r++) {
    const file1 = TOWN_BUILDING_FILES[(r * 2) % TOWN_BUILDING_FILES.length];
    const file2 = TOWN_BUILDING_FILES[(r * 2 + 1) % TOWN_BUILDING_FILES.length];

    // Left side (west, -X)
    townBuildings.push({
      x: -BUILDING_OFFSET,
      z: -r * TILE_SIZE,
      rotation: Math.PI / 2,
      modelFile: file1,
    });
    // Right side (east, +X)
    townBuildings.push({
      x: BUILDING_OFFSET,
      z: -r * TILE_SIZE,
      rotation: -Math.PI / 2,
      modelFile: file2,
    });
  }

  // ---- Props along financial district road ----
  for (let c = 1; c <= bankRows; c++) {
    const propName = PROP_CYCLE[c % PROP_CYCLE.length];
    props.push({
      x: c * TILE_SIZE + TILE_SIZE * 0.4,
      z: -PROP_OFFSET,
      rotation: 0,
      propType: propName,
    });
    props.push({
      x: c * TILE_SIZE + TILE_SIZE * 0.4,
      z: PROP_OFFSET,
      rotation: Math.PI,
      propType: propName === 'streetlight' ? 'streetlight' : 'bush',
    });
  }

  // ---- Props along residential road ----
  for (let r = 1; r <= RESIDENTIAL_ROWS; r++) {
    const propName = PROP_CYCLE[r % PROP_CYCLE.length];
    props.push({
      x: -PROP_OFFSET,
      z: -r * TILE_SIZE + TILE_SIZE * 0.4,
      rotation: 0,
      propType: propName,
    });
    props.push({
      x: PROP_OFFSET,
      z: -r * TILE_SIZE + TILE_SIZE * 0.4,
      rotation: Math.PI,
      propType: propName === 'streetlight' ? 'streetlight' : 'bush',
    });
  }

  // ---- Parked vehicles in residential zone for ambiance ----
  for (let r = 1; r <= RESIDENTIAL_ROWS; r += 2) {
    vehicles.push({
      x: PROP_OFFSET + 0.6,
      z: -r * TILE_SIZE + 0.5,
      rotation: 0,
      modelFile: 'vehicle.gltf',
    });
  }

  // Key positions
  const spawnPoint = { x: 0, z: -(RESIDENTIAL_ROWS + 0.5) * TILE_SIZE };
  const financialCenter = { x: (bankRows * TILE_SIZE) / 2 + TILE_SIZE / 2, z: 0 };
  const residentialCenter = { x: 0, z: -(RESIDENTIAL_ROWS * TILE_SIZE) / 2 };

  return {
    buildings,
    townBuildings,
    roads,
    props,
    vehicles,
    tileSize: TILE_SIZE,
    totalLength: bankRows * TILE_SIZE,        // financial district length
    residentialLength: RESIDENTIAL_ROWS * TILE_SIZE,
    buildingOffset: BUILDING_OFFSET,
    spawnPoint,
    financialCenter,
    residentialCenter,
  };
}
