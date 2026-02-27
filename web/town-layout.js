/**
 * Pure data layout algorithm for the town.
 * Financial district sits on +X. Borrowers live in an industrial suburb on -X/-Z
 * connected by a bridge into the core junction at (0,0,0).
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

const INDUSTRIAL_BUILDING_FILES = [
  'building_H.gltf', 'building_G.gltf', 'building_F.gltf', 'building_E.gltf',
  'building_D.gltf', 'building_C.gltf', 'building_B.gltf', 'building_A.gltf',
];

const PROP_CYCLE = ['streetlight', 'bench', 'bush', 'firehydrant', 'bush', 'bench'];

const BORROWER_ROWS = 4;  // industrial suburb rows
const BRIDGE_TILES = 3;

const HEX_SIZE = 0.52;
const HEX_CLUSTER_RADIUS_BASE = 4.6;

export function generateLayout(lenderCount, buildingFiles) {
  const BUILDING_FILES = buildingFiles || DEFAULT_BUILDING_FILES;
  const bankRows = Math.ceil(lenderCount / 2);
  const roads = [];
  const buildings = [];   // bank buildings (lender-indexed)
  const townBuildings = []; // decorative central buildings
  const borrowerDistrictBuildings = []; // industrial borrower suburb
  const bridgeRoads = [];
  const props = [];
  const vehicles = [];
  const borrowerDistrictRoads = [];

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

  // ---- Borrower suburb: industrial road grid on -X/-Z ----
  const suburbBaseX = -8;
  const suburbBaseZ = -8;
  for (let c = 0; c < BORROWER_ROWS; c++) {
    const x = suburbBaseX - c * TILE_SIZE;
    borrowerDistrictRoads.push({
      x, z: suburbBaseZ, rotation: Math.PI / 2,
      modelFile: 'road_straight.gltf',
    });
    roads.push(borrowerDistrictRoads[borrowerDistrictRoads.length - 1]);
  }
  for (let r = 1; r <= BORROWER_ROWS; r++) {
    const z = suburbBaseZ + r * TILE_SIZE;
    borrowerDistrictRoads.push({
      x: suburbBaseX, z, rotation: 0,
      modelFile: r === BORROWER_ROWS ? 'road_tsplit.gltf' : 'road_straight.gltf',
    });
    roads.push(borrowerDistrictRoads[borrowerDistrictRoads.length - 1]);
  }

  // ---- Bridge from suburb to core junction ----
  for (let b = 1; b <= BRIDGE_TILES; b++) {
    const t = b / (BRIDGE_TILES + 1);
    const x = suburbBaseX + (0 - suburbBaseX) * t;
    const z = suburbBaseZ + (0 - suburbBaseZ) * t;
    const seg = {
      x,
      z,
      rotation: Math.PI / 4,
      modelFile: 'road_straight.gltf',
      isBridge: true,
    };
    bridgeRoads.push(seg);
    roads.push(seg);
  }

  // ---- Industrial buildings in borrower suburb ----
  for (let r = 0; r < BORROWER_ROWS; r++) {
    const file1 = INDUSTRIAL_BUILDING_FILES[(r * 2) % INDUSTRIAL_BUILDING_FILES.length];
    const file2 = INDUSTRIAL_BUILDING_FILES[(r * 2 + 1) % INDUSTRIAL_BUILDING_FILES.length];
    const z = suburbBaseZ + r * TILE_SIZE;
    borrowerDistrictBuildings.push({
      x: suburbBaseX - BUILDING_OFFSET + 0.8,
      z,
      rotation: Math.PI / 2,
      modelFile: file1,
      zone: 'borrower',
    });
    borrowerDistrictBuildings.push({
      x: suburbBaseX + BUILDING_OFFSET - 0.8,
      z,
      rotation: -Math.PI / 2,
      modelFile: file2,
      zone: 'borrower',
    });
  }

  // ---- Decorative central town buildings ----
  for (let r = 1; r <= 3; r++) {
    const file1 = TOWN_BUILDING_FILES[(r * 2) % TOWN_BUILDING_FILES.length];
    const file2 = TOWN_BUILDING_FILES[(r * 2 + 1) % TOWN_BUILDING_FILES.length];
    townBuildings.push({
      x: -BUILDING_OFFSET,
      z: r * TILE_SIZE,
      rotation: Math.PI / 2,
      modelFile: file1,
    });
    townBuildings.push({
      x: BUILDING_OFFSET,
      z: r * TILE_SIZE,
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

  // ---- Props in borrower suburb ----
  for (let r = 1; r <= BORROWER_ROWS; r++) {
    const propName = PROP_CYCLE[r % PROP_CYCLE.length];
    props.push({
      x: suburbBaseX - PROP_OFFSET,
      z: suburbBaseZ + r * TILE_SIZE - TILE_SIZE * 0.4,
      rotation: 0,
      propType: propName,
    });
    props.push({
      x: suburbBaseX + PROP_OFFSET,
      z: suburbBaseZ + r * TILE_SIZE - TILE_SIZE * 0.4,
      rotation: Math.PI,
      propType: propName === 'streetlight' ? 'streetlight' : 'bush',
    });
  }

  // ---- Parked vehicles in borrower suburb for ambiance ----
  for (let r = 1; r <= BORROWER_ROWS; r += 2) {
    vehicles.push({
      x: suburbBaseX + PROP_OFFSET + 0.3,
      z: suburbBaseZ + r * TILE_SIZE - 0.4,
      rotation: 0,
      modelFile: 'vehicle.gltf',
    });
  }

  // Key positions
  const spawnPoint = { x: suburbBaseX - 1.2, z: suburbBaseZ - 0.6 };
  const suburbEntry = { x: suburbBaseX, z: suburbBaseZ };
  const bridgeMidpoint = bridgeRoads.length
    ? { x: bridgeRoads[Math.floor(bridgeRoads.length / 2)].x, z: bridgeRoads[Math.floor(bridgeRoads.length / 2)].z }
    : { x: -4, z: -4 };
  const financialCenter = { x: (bankRows * TILE_SIZE) / 2 + TILE_SIZE / 2, z: 0 };
  const residentialCenter = { x: suburbBaseX - 1, z: suburbBaseZ + TILE_SIZE };

  return {
    type: 'town',
    buildings,
    townBuildings,
    borrowerDistrictBuildings,
    bridgeRoads,
    borrowerDistrictRoads,
    roads,
    props,
    vehicles,
    tileSize: TILE_SIZE,
    totalLength: bankRows * TILE_SIZE,        // financial district length
    residentialLength: BORROWER_ROWS * TILE_SIZE,
    buildingOffset: BUILDING_OFFSET,
    spawnPoint,
    suburbEntry,
    bridgeMidpoint,
    financialCenter,
    residentialCenter,
  };
}

function _axialToWorld(q, r, size = HEX_SIZE) {
  return {
    x: size * 1.5 * q,
    z: size * Math.sqrt(3) * (r + q / 2),
  };
}

function _hexSpiral(count) {
  if (count <= 0) return [];
  const out = [];
  const dirs = [
    [1, 0], [0, 1], [-1, 1],
    [-1, 0], [0, -1], [1, -1],
  ];
  let radius = 1;
  while (out.length < count) {
    let q = radius;
    let r = 0;
    for (let side = 0; side < 6; side++) {
      const [dq, dr] = dirs[side];
      for (let step = 0; step < radius; step++) {
        out.push({ q, r });
        if (out.length >= count) return out;
        q += dq;
        r += dr;
      }
    }
    radius += 1;
  }
  return out;
}

function _portfolioWeights(lenders = [], shareMode = 'capital') {
  if (shareMode === 'equal') {
    return lenders.map(() => 1);
  }
  if (shareMode === 'capital-soft') {
    return lenders.map(l => Math.max(1, Math.sqrt(Math.max(1, Number(l?.total_capital) || 1))));
  }
  if (shareMode === 'max-loan') {
    return lenders.map(l => Math.max(1, Number(l?.max_single_loan) || 1));
  }
  // Default: capital-proportional.
  return lenders.map(l => Math.max(1, Number(l?.total_capital) || 1));
}

function _hexRingForSlots(slotCount) {
  if (slotCount <= 0) return 0;
  // Slots are generated ring-by-ring around center, excluding center cell:
  // total slots up to ring r = 3 * r * (r + 1)
  return Math.ceil((-1 + Math.sqrt(1 + (4 * slotCount) / 3)) / 2);
}

function _portfolioShareCounts(lenders = [], totalPotentialBorrowers = 0, shareMode = 'capital') {
  const lenderCount = lenders.length;
  if (!lenderCount) return [];

  const target = Math.max(lenderCount, Math.round(totalPotentialBorrowers || 0));
  const weights = _portfolioWeights(lenders, shareMode);
  const totalWeight = weights.reduce((acc, v) => acc + v, 0) || lenderCount;
  const exact = weights.map(w => (w / totalWeight) * target);
  const counts = exact.map(v => Math.floor(v));
  const remainders = exact.map((v, i) => ({ i, frac: v - counts[i] }))
    .sort((a, b) => b.frac - a.frac);

  let assigned = counts.reduce((acc, v) => acc + v, 0);

  // Every lender should have at least one portfolio hex.
  for (let i = 0; i < lenderCount; i++) {
    if (counts[i] < 1) {
      counts[i] = 1;
      assigned += 1;
    }
  }

  let cursor = 0;
  while (assigned < target) {
    const idx = remainders[cursor % remainders.length].i;
    counts[idx] += 1;
    assigned += 1;
    cursor += 1;
  }

  // Trim over-allocation while preserving at least one slot per lender.
  const reverse = [...remainders].reverse();
  cursor = 0;
  while (assigned > target) {
    const idx = reverse[cursor % reverse.length].i;
    if (counts[idx] > 1) {
      counts[idx] -= 1;
      assigned -= 1;
    }
    cursor += 1;
    if (cursor > reverse.length * 4) break;
  }

  return counts;
}

export function generatePortfolioLayout(lenders = [], seasonWeeks = [], options = {}) {
  const shareMode = options.shareMode || 'capital';
  const lenderCount = lenders.length;
  const totalPotentialBorrowers = (seasonWeeks || [])
    .filter(w => !w?.final_resolution)
    .reduce((acc, week) => acc + ((week?.borrowers || []).length), 0);
  const shareCounts = _portfolioShareCounts(lenders, totalPotentialBorrowers, shareMode);

  const buildings = [];
  const portfolioSlots = [];
  const portfolioSlotsByLender = {};
  const maxRing = shareCounts.reduce((acc, c) => Math.max(acc, _hexRingForSlots(c)), 1);
  const clusterFootprint = maxRing * HEX_SIZE * 1.95 + 0.78; // lender cluster footprint
  const minCenterDistance = clusterFootprint * 2 + 0.35;
  const radiusFromArc = lenderCount > 1
    ? (minCenterDistance / (2 * Math.sin(Math.PI / lenderCount)))
    : 0;
  const clusterRadius = Math.max(HEX_CLUSTER_RADIUS_BASE, lenderCount * 1.1, radiusFromArc);

  for (let lenderIndex = 0; lenderIndex < lenderCount; lenderIndex++) {
    const angle = (Math.PI * 2 * lenderIndex) / Math.max(1, lenderCount) - Math.PI / 2;
    const centerX = Math.cos(angle) * clusterRadius;
    const centerZ = Math.sin(angle) * clusterRadius;
    const slotCount = Math.max(1, shareCounts[lenderIndex] || 1);
    const axialSlots = _hexSpiral(slotCount);

    buildings.push({
      id: lenderIndex,
      x: centerX,
      z: centerZ,
      rotation: 0,
      modelFile: '',
    });

    const lenderSlots = [];
    for (let slotIndex = 0; slotIndex < axialSlots.length; slotIndex++) {
      const axial = axialSlots[slotIndex];
      const local = _axialToWorld(axial.q, axial.r);
      const slot = {
        lenderIndex,
        slotIndex,
        x: centerX + local.x,
        z: centerZ + local.z,
      };
      portfolioSlots.push(slot);
      lenderSlots.push(slot);
    }
    portfolioSlotsByLender[lenderIndex] = lenderSlots;
  }

  return {
    type: 'portfolio-hex',
    shareMode,
    buildings,
    roads: [],
    props: [],
    vehicles: [],
    townBuildings: [],
    borrowerDistrictBuildings: [],
    bridgeRoads: [],
    borrowerDistrictRoads: [],
    portfolioSlots,
    portfolioSlotsByLender,
    hexRadius: HEX_SIZE,
    tileSize: 2,
    totalLength: 0,
    residentialLength: 0,
    buildingOffset: 0,
    spawnPoint: { x: 0, z: -13.5 },
    suburbEntry: { x: 0, z: -8.5 },
    bridgeMidpoint: { x: 0, z: -4.0 },
    financialCenter: { x: 0, z: 0 },
    residentialCenter: { x: 0, z: 0 },
  };
}
