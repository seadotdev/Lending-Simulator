import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

export class TownScene {
  constructor(options = {}) {
    this.visualizationMode = options.mode || 'town';
    this.renderer = null;
    this.labelRenderer = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
    this.clock = new THREE.Clock();

    this.buildingMeshes = new Map();  // lenderIndex → Group
    this.buildingLabels = new Map();  // lenderIndex → CSS2DObject
    this.buildingStatusBadges = new Map(); // lenderIndex → CSS2DObject
    this.buildingSmokeBadges = new Map();  // lenderIndex → CSS2DObject
    this.portfolioHexMeshes = new Map();   // lenderIndex → Mesh[]
    this.borrowerMeshes = new Map();  // borrowerId → Group
    this.borrowerLabels = new Map();  // borrowerId → CSS2DObject
    this.borrowerStates = new Map();  // borrowerId → state string
    this.mixers = [];                 // AnimationMixer[]
    this.tweens = [];                 // active tweens
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.hoveredBuilding = null;
    this.selectedBuilding = null;
    this.characterAssetsUnavailable = false;
    this.animationFrame = null;
    this.disposed = false;
    this._boundPointerMove = null;
    this._boundClick = null;
    this._boundResize = null;

    this.onBuildingClick = null;      // callback(lenderIndex)
    this.onBuildingHover = null;      // callback(lenderIndex|null)

    this._animate = this._animate.bind(this);
  }

  init(container) {
    this.container = container;
    this.disposed = false;
    const w = container.clientWidth;
    const h = container.clientHeight;

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    // CSS2D label renderer
    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(w, h);
    this.labelRenderer.domElement.style.position = 'absolute';
    this.labelRenderer.domElement.style.top = '0';
    this.labelRenderer.domElement.style.left = '0';
    this.labelRenderer.domElement.style.pointerEvents = 'none';
    container.appendChild(this.labelRenderer.domElement);

    // Scene
    this.scene = new THREE.Scene();
    const bgColor = this.visualizationMode === 'portfolio-hex' ? 0x101621 : 0x0f1520;
    this.scene.background = new THREE.Color(bgColor);
    this.scene.fog = new THREE.Fog(bgColor, 30, 70);

    // Camera (isometric-ish)
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 150);
    this.camera.position.set(12, 14, 8);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.maxPolarAngle = Math.PI / 2.2;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 40;

    // Lights
    const ambient = new THREE.AmbientLight(0x404060, this.visualizationMode === 'portfolio-hex' ? 0.72 : 0.6);
    this.scene.add(ambient);

    const hemi = new THREE.HemisphereLight(0x88aacc, 0x443322, this.visualizationMode === 'portfolio-hex' ? 0.55 : 0.5);
    this.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffeedd, this.visualizationMode === 'portfolio-hex' ? 1.0 : 1.2);
    dir.position.set(6, 12, 8);
    dir.castShadow = true;
    dir.shadow.mapSize.set(2048, 2048);
    dir.shadow.camera.near = 0.5;
    dir.shadow.camera.far = 40;
    dir.shadow.camera.left = -15;
    dir.shadow.camera.right = 15;
    dir.shadow.camera.top = 15;
    dir.shadow.camera.bottom = -15;
    this.scene.add(dir);

    // Ground plane with grid texture
    const groundGeo = new THREE.PlaneGeometry(100, 100);
    const gridCanvas = document.createElement('canvas');
    gridCanvas.width = 512;
    gridCanvas.height = 512;
    const ctx = gridCanvas.getContext('2d');
    ctx.fillStyle = this.visualizationMode === 'portfolio-hex' ? '#141e2d' : '#1a3320';
    ctx.fillRect(0, 0, 512, 512);
    if (this.visualizationMode === 'portfolio-hex') {
      const hexR = 16;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.055)';
      ctx.lineWidth = 1;
      for (let row = -4; row < 28; row++) {
        for (let col = -4; col < 28; col++) {
          const cx = col * hexR * 1.5 + ((row % 2) ? hexR * 0.75 : 0);
          const cy = row * hexR * Math.sqrt(3) / 2;
          ctx.beginPath();
          for (let i = 0; i < 6; i++) {
            const a = Math.PI / 6 + i * Math.PI / 3;
            const x = cx + Math.cos(a) * hexR;
            const y = cy + Math.sin(a) * hexR;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();
          ctx.stroke();
        }
      }
    } else {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.lineWidth = 1;
      const gridStep = 32;
      for (let x = 0; x <= 512; x += gridStep) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 512); ctx.stroke();
      }
      for (let y = 0; y <= 512; y += gridStep) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(512, y); ctx.stroke();
      }
    }
    const gridTexture = new THREE.CanvasTexture(gridCanvas);
    gridTexture.wrapS = THREE.RepeatWrapping;
    gridTexture.wrapT = THREE.RepeatWrapping;
    gridTexture.repeat.set(8, 8);
    const groundMat = new THREE.MeshStandardMaterial({ map: gridTexture, roughness: 0.9 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Events
    this._boundPointerMove = (e) => this._onPointerMove(e);
    this._boundClick = (e) => this._onClick(e);
    this._boundResize = () => this._onResize();
    this.renderer.domElement.addEventListener('pointermove', this._boundPointerMove);
    this.renderer.domElement.addEventListener('click', this._boundClick);
    window.addEventListener('resize', this._boundResize);

    this._animate();
  }

  async buildTown(layout, lenderNames, assetLoader) {
    this.layout = layout;
    this.buildingMeshes.clear();
    this.buildingLabels.clear();
    this.buildingStatusBadges.clear();
    this.buildingSmokeBadges.clear();
    this.portfolioHexMeshes.clear();
    this.borrowerMeshes.clear();
    this.borrowerLabels.clear();
    this.borrowerStates.clear();
    const activePack = assetLoader?.manifest?.activePack || null;
    const packConfig = assetLoader?.getActivePack?.() || null;
    this.characterAssetsUnavailable = false;
    let packAssetsUnavailable = !activePack;
    let fallbackUsed = false;

    if (layout?.type === 'portfolio-hex') {
      this._buildPortfolioHex(layout, lenderNames || []);
      return { fallbackUsed: false };
    }

    // Center camera to see both zones (financial along +X, residential along -Z)
    const fc = layout.financialCenter || { x: 4, z: 0 };
    const rc = layout.residentialCenter || { x: 0, z: -4 };
    const cx = (fc.x + rc.x) / 2;
    const cz = (fc.z + rc.z) / 2;
    this.controls.target.set(cx, 0, cz);
    this.camera.position.set(cx + 12, 14, cz + 12);

    // Roads
    for (const road of layout.roads) {
      let model = null;
      if (!packAssetsUnavailable) {
        try {
          const loaded = await assetLoader.loadModel(activePack, road.modelFile);
          model = loaded.scene;
        } catch (e) {
          packAssetsUnavailable = true;
          fallbackUsed = true;
          console.warn('Asset pack unavailable, using fallback geometry:', e);
        }
      }
      if (!model) {
        model = this._createFallbackRoad(road);
      }
      model.position.set(road.x, road.isBridge ? 0.25 : 0, road.z);
      model.rotation.y = road.rotation;
      model.traverse(c => { if (c.isMesh) { c.receiveShadow = true; } });
      this.scene.add(model);
    }

    // Bank buildings (lender-indexed)
    for (const bld of layout.buildings) {
      let model = null;
      if (!packAssetsUnavailable) {
        try {
          const loaded = await assetLoader.loadModel(activePack, bld.modelFile);
          model = loaded.scene;
        } catch (e) {
          packAssetsUnavailable = true;
          fallbackUsed = true;
          console.warn('Failed to load building assets, using fallback geometry:', e);
        }
      }
      if (!model) {
        model = this._createFallbackBuilding(bld.id);
      }

      model.position.set(bld.x, 0, bld.z);
      model.rotation.y = bld.rotation;
      model.traverse(c => {
        if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; }
      });
      model.userData.lenderIndex = bld.id;
      this.buildingMeshes.set(bld.id, model);
      this.scene.add(model);

      // Signpost label
      const name = lenderNames[bld.id] || `Lender ${bld.id}`;
      const labelDiv = document.createElement('div');
      labelDiv.className = 'bank-signpost';
      labelDiv.innerHTML = this._signpostHTML(name);
      labelDiv.addEventListener('click', (e) => {
        e.stopPropagation();
        this._toggleSignpost(labelDiv);
      });
      const label = new CSS2DObject(labelDiv);
      label.position.set(0, 2.5, 0);
      model.add(label);
      this.buildingLabels.set(bld.id, label);

      const statusDiv = document.createElement('div');
      statusDiv.className = 'lender-status-badge';
      const status = new CSS2DObject(statusDiv);
      status.position.set(0, 4.1, 0);
      model.add(status);
      this.buildingStatusBadges.set(bld.id, status);

      const smokeDiv = document.createElement('div');
      smokeDiv.className = 'lender-smoke';
      smokeDiv.textContent = '';
      const smoke = new CSS2DObject(smokeDiv);
      smoke.position.set(0.3, 3.6, 0.1);
      model.add(smoke);
      this.buildingSmokeBadges.set(bld.id, smoke);
    }

    // Decorative town buildings (residential zone)
    if (layout.townBuildings) {
      for (const bld of layout.townBuildings) {
        let model = null;
        if (!packAssetsUnavailable) {
          try {
            const loaded = await assetLoader.loadModel(activePack, bld.modelFile);
            model = loaded.scene;
          } catch (e) {
            packAssetsUnavailable = true;
            fallbackUsed = true;
          }
        }
        if (!model) {
          model = this._createFallbackTownBuilding();
        }
        model.position.set(bld.x, 0, bld.z);
        model.rotation.y = bld.rotation;
        model.traverse(c => {
          if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; }
        });
        this.scene.add(model);
      }
    }

    // Borrower suburb buildings (industrial fallback style)
    if (layout.borrowerDistrictBuildings) {
      for (const bld of layout.borrowerDistrictBuildings) {
        let model = null;
        if (!packAssetsUnavailable) {
          try {
            const loaded = await assetLoader.loadModel(activePack, bld.modelFile);
            model = loaded.scene;
          } catch (e) {
            packAssetsUnavailable = true;
            fallbackUsed = true;
          }
        }
        if (!model) {
          model = this._createFallbackIndustrialBuilding();
        }
        model.position.set(bld.x, 0, bld.z);
        model.rotation.y = bld.rotation;
        model.traverse(c => {
          if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; }
        });
        this.scene.add(model);
      }
    }

    // Vehicles (parked cars in residential zone)
    if (layout.vehicles) {
      for (const veh of layout.vehicles) {
        let model = null;
        if (!packAssetsUnavailable) {
          try {
            const loaded = await assetLoader.loadModel(activePack, veh.modelFile);
            model = loaded.scene;
          } catch (e) {
            packAssetsUnavailable = true;
            fallbackUsed = true;
          }
        }
        if (!model) {
          model = this._createFallbackVehicle();
        }
        model.position.set(veh.x, 0, veh.z);
        model.rotation.y = veh.rotation;
        model.traverse(c => { if (c.isMesh) { c.castShadow = true; } });
        this.scene.add(model);
      }
    }

    // Props
    for (const prop of layout.props) {
      const propFile = packConfig?.props?.[prop.propType];
      let model = null;
      if (propFile && !packAssetsUnavailable) {
        try {
          const loaded = await assetLoader.loadModel(activePack, propFile);
          model = loaded.scene;
        } catch (e) {
          packAssetsUnavailable = true;
          fallbackUsed = true;
        }
      }
      if (!model) {
        model = this._createFallbackProp(prop.propType);
      }
      model.position.set(prop.x, 0, prop.z);
      model.rotation.y = prop.rotation;
      model.traverse(c => { if (c.isMesh) { c.castShadow = true; } });
      this.scene.add(model);
    }

    if (packAssetsUnavailable) {
      this.characterAssetsUnavailable = true;
    }
    return { fallbackUsed };
  }

  _buildPortfolioHex(layout, lenderNames) {
    const RISE_DEPTH = -10;
    const centers = layout?.buildings || [];
    const maxRadius = centers.reduce((acc, b) => Math.max(acc, Math.hypot(b.x, b.z)), 8);
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(0, Math.max(13, maxRadius * 1.75), Math.max(11, maxRadius * 1.55));

    // 1. Landscape terrain rises from below
    const landscapeTiles = this._buildHexWorldMap(layout);
    for (const tile of landscapeTiles) tile.mesh.position.y = RISE_DEPTH;

    // 2. Lender cores — start hidden above, will drop with a thud
    const DROP_HEIGHT = 14;
    const coreTiles = [];
    for (const bld of centers) {
      const lenderIndex = bld.id;
      const core = this._createPortfolioCore(lenderIndex);
      core.position.set(bld.x, DROP_HEIGHT, bld.z);
      core.userData.lenderIndex = lenderIndex;
      this.scene.add(core);
      this.buildingMeshes.set(lenderIndex, core);
      this.portfolioHexMeshes.set(lenderIndex, []);
      coreTiles.push({ mesh: core, wx: bld.x, wz: bld.z, lenderIndex });

      const name = lenderNames[lenderIndex] || `Lender ${lenderIndex}`;
      const slotCount = (layout?.portfolioSlotsByLender?.[lenderIndex] || []).length;
      const labelDiv = document.createElement('div');
      labelDiv.className = 'bank-signpost portfolio-signpost';
      labelDiv.innerHTML = this._signpostHTML(name, slotCount);
      labelDiv.addEventListener('click', (e) => {
        e.stopPropagation();
        this._toggleSignpost(labelDiv);
      });
      const label = new CSS2DObject(labelDiv);
      label.position.set(0, 2.2, 0);
      core.add(label);
      this.buildingLabels.set(lenderIndex, label);

      const statusDiv = document.createElement('div');
      statusDiv.className = 'lender-status-badge';
      const status = new CSS2DObject(statusDiv);
      status.position.set(0, 3.45, 0);
      core.add(status);
      this.buildingStatusBadges.set(lenderIndex, status);

      const smokeDiv = document.createElement('div');
      smokeDiv.className = 'lender-smoke';
      smokeDiv.textContent = '';
      const smoke = new CSS2DObject(smokeDiv);
      smoke.position.set(0.28, 2.95, 0.12);
      core.add(smoke);
      this.buildingSmokeBadges.set(lenderIndex, smoke);
    }

    // 3. Portfolio hex cells — stay underground, rise after their lender lands
    const hexRadius = layout?.hexRadius || 0.52;
    const portfolioTilesByLender = new Map();
    for (const slot of layout?.portfolioSlots || []) {
      const hex = this._createPortfolioHexCell(slot.lenderIndex, hexRadius);
      hex.position.set(slot.x, RISE_DEPTH, slot.z);
      hex.userData.lenderIndex = slot.lenderIndex;
      hex.userData.slotIndex = slot.slotIndex;
      this.scene.add(hex);
      const lenderSlots = this.portfolioHexMeshes.get(slot.lenderIndex) || [];
      lenderSlots.push(hex);
      this.portfolioHexMeshes.set(slot.lenderIndex, lenderSlots);
      if (!portfolioTilesByLender.has(slot.lenderIndex)) portfolioTilesByLender.set(slot.lenderIndex, []);
      portfolioTilesByLender.get(slot.lenderIndex).push({ mesh: hex, finalY: 0.08, wx: slot.x, wz: slot.z });
    }

    // Sequence: landscape rises → lenders drop with thuds → territory hexes rise
    this._scheduleHexMapReveal(landscapeTiles, coreTiles, portfolioTilesByLender, layout);
  }

  // ---- Hex landscape terrain generation ----

  _hashCoord(a, b) {
    a = a | 0; b = b | 0;
    let h = (Math.imul(a, 0x9e3779b9) ^ Math.imul(b, 0x6c62272e)) >>> 0;
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0;
    h ^= h >>> 15;
    return (h >>> 0) / 4294967295;
  }

  _landscapeTerrainType(q, r) {
    // Multi-scale noise creates natural biome patches
    const c1 = this._hashCoord(Math.floor(q / 6), Math.floor(r / 6));
    const c2 = this._hashCoord(Math.floor(q / 3) + 50, Math.floor(r / 3) + 50);
    const fine = this._hashCoord(q + 200, r + 200);
    const noise = c1 * 0.60 + c2 * 0.25 + fine * 0.15;
    if (noise < 0.13) return 'water';
    if (noise < 0.45) return 'grass';
    if (noise < 0.68) return 'forest';
    if (noise < 0.84) return 'hill';
    return 'rock';
  }

  _buildHexWorldMap(layout) {
    const hexRadius = layout.hexRadius || 0.52;
    const tileSize = hexRadius * 2.0;  // landscape tiles are 2x portfolio hex size
    const clusterExtent = (layout.buildings || []).reduce(
      (acc, b) => Math.max(acc, Math.hypot(b.x, b.z)), 8
    );
    const worldExtent = clusterExtent + 7;
    const gridR = Math.ceil(worldExtent / (tileSize * Math.sqrt(3))) + 1;

    const portfolioPositions = (layout.portfolioSlots || []).map(s => [s.x, s.z]);
    const corePositions = (layout.buildings || []).map(b => [b.x, b.z]);
    const portfolioClear = tileSize * 0.9;
    const coreClear = tileSize * 1.5;

    const tiles = [];
    for (let q = -gridR; q <= gridR; q++) {
      for (let r = -gridR; r <= gridR; r++) {
        const s = -q - r;
        if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) > gridR) continue;

        const wx = tileSize * 1.5 * q;
        const wz = tileSize * Math.sqrt(3) * (r + q / 2);

        // Skip positions that overlap portfolio cluster cells or cores
        let blocked = false;
        for (const [px, pz] of portfolioPositions) {
          if (Math.hypot(wx - px, wz - pz) < portfolioClear) { blocked = true; break; }
        }
        if (!blocked) {
          for (const [cx, cz] of corePositions) {
            if (Math.hypot(wx - cx, wz - cz) < coreClear) { blocked = true; break; }
          }
        }
        if (blocked) continue;

        const type = this._landscapeTerrainType(q, r);
        const mesh = this._createTerrainHex(type, tileSize, q, r);
        mesh.position.set(wx, 0, wz);
        this.scene.add(mesh);
        tiles.push({ mesh, finalY: 0, wx, wz });
      }
    }
    return tiles;
  }

  _createTerrainHex(type, radius, q, r) {
    const group = new THREE.Group();
    const rng = (n) => this._hashCoord(q * 37 + n * 7, r * 53 + n * 13);

    const HEIGHT = { water: 0.07, grass: 0.15, forest: 0.18, hill: 0.32, rock: 0.55 };
    const BASE_COLOR = {
      water: 0x1b4f72, grass: 0x3d6b32, forest: 0x2a5124, hill: 0x7a6a4c, rock: 0x52525e,
    };

    const h = HEIGHT[type] ?? 0.15;
    const col = new THREE.Color(BASE_COLOR[type] ?? 0x3d6b32);
    // Subtle color variation per tile
    col.r = Math.max(0, Math.min(1, col.r + (rng(1) - 0.5) * 0.06));
    col.g = Math.max(0, Math.min(1, col.g + (rng(2) - 0.5) * 0.06));
    col.b = Math.max(0, Math.min(1, col.b + (rng(3) - 0.5) * 0.04));

    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.95, radius, h, 6),
      new THREE.MeshStandardMaterial({
        color: col,
        roughness: type === 'water' ? 0.15 : 0.88,
        metalness: type === 'water' ? 0.12 : 0.04,
      })
    );
    base.rotation.y = Math.PI / 6;
    base.position.y = h / 2;
    base.receiveShadow = true;
    base.castShadow = type !== 'water';
    group.add(base);

    if (type === 'forest') {
      const count = 2 + Math.floor(rng(4) * 3);
      for (let i = 0; i < count; i++) {
        const angle = rng(i * 3 + 5) * Math.PI * 2;
        const dist = rng(i * 3 + 6) * radius * 0.52;
        const tH = 0.32 + rng(i * 3 + 7) * 0.38;
        const tR = 0.07 + rng(i * 3 + 8) * 0.06;
        const treeCol = new THREE.Color(0x2d4f24);
        treeCol.g += rng(i + 10) * 0.12;
        const tree = new THREE.Mesh(
          new THREE.ConeGeometry(tR, tH, 6),
          new THREE.MeshStandardMaterial({ color: treeCol, roughness: 0.9 })
        );
        tree.position.set(Math.cos(angle) * dist, h + tH / 2, Math.sin(angle) * dist);
        tree.castShadow = true;
        group.add(tree);
      }
    } else if (type === 'hill') {
      const cap = new THREE.Mesh(
        new THREE.ConeGeometry(radius * 0.38, h * 0.55, 6),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(0x8f7c60), roughness: 0.92 })
      );
      cap.rotation.y = Math.PI / 6;
      cap.position.y = h + h * 0.28;
      cap.castShadow = true;
      group.add(cap);
    } else if (type === 'rock') {
      const peak = new THREE.Mesh(
        new THREE.ConeGeometry(radius * 0.42, h * 0.65, 5),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(0x666670), roughness: 0.85 })
      );
      peak.rotation.y = Math.PI / 7 + rng(9) * 0.4;
      peak.position.y = h + h * 0.32;
      peak.castShadow = true;
      group.add(peak);
      // Snow cap
      const snow = new THREE.Mesh(
        new THREE.ConeGeometry(radius * 0.16, h * 0.18, 5),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(0xe0e4ea), roughness: 0.95 })
      );
      snow.position.y = h + h * 0.32 + h * 0.42;
      group.add(snow);
    }
    return group;
  }

  _scheduleHexMapReveal(landscapeTiles, coreTiles, portfolioTilesByLender, layout) {
    const RISE_DEPTH = -10;
    const WAVE_SPEED = 0.10;   // s per world unit — slower, more dramatic
    const RISE_DURATION = 0.90;
    const BASE_DELAY = 0.15;

    // 1. Landscape tiles rise from below in an outward wave
    let maxLandscapeDelay = BASE_DELAY;
    for (const tile of landscapeTiles) {
      const dist = Math.hypot(tile.wx || 0, tile.wz || 0);
      const jitter = (this._hashCoord(
        Math.round((tile.wx || 0) * 10),
        Math.round((tile.wz || 0) * 10)
      ) - 0.5) * 0.12;
      const delay = Math.max(0, BASE_DELAY + dist * WAVE_SPEED + jitter);
      maxLandscapeDelay = Math.max(maxLandscapeDelay, delay);

      const { mesh, finalY } = tile;
      const startPos = new THREE.Vector3(mesh.position.x, RISE_DEPTH, mesh.position.z);
      const endPos = new THREE.Vector3(mesh.position.x, finalY ?? 0, mesh.position.z);

      setTimeout(() => {
        if (this.disposed) return;
        this.tweens.push({ elapsed: 0, duration: RISE_DURATION, startPos, endPos, mesh, resolve: () => {} });
      }, delay * 1000);
    }

    // 2. After landscape settles, drop lender cores one by one with a thud
    const LENDER_INTERVAL = 0.55; // s between each lender landing
    const coresStart = maxLandscapeDelay + RISE_DURATION + 0.35;

    coreTiles.forEach((core, idx) => {
      const dropTime = coresStart + idx * LENDER_INTERVAL;
      setTimeout(() => {
        if (this.disposed) return;
        this._animateLenderDrop(core.mesh, core.wx, core.wz, () => {
          // 3. After lender lands, portfolio hexes ripple outward from core
          const pTiles = portfolioTilesByLender?.get(core.lenderIndex) || [];
          this._schedulePortfolioRise(pTiles, core.wx, core.wz);
        });
      }, dropTime * 1000);
    });
  }

  _animateLenderDrop(mesh, wx, wz, onLanded) {
    const DROP_HEIGHT = mesh.position.y; // starts above ground
    const LAND_Y = 0.0;
    const DROP_MS = 560;
    const BOUNCE_HEIGHT = 0.50;
    const BOUNCE_MS = 220;

    const start = performance.now();
    const animate = (now) => {
      if (this.disposed) return;
      const t = Math.min((now - start) / DROP_MS, 1);
      // Ease-in-cubic (gravity feel)
      const eased = t * t * t;
      mesh.position.y = DROP_HEIGHT + (LAND_Y - DROP_HEIGHT) * eased;

      if (t < 1) {
        requestAnimationFrame(animate);
      } else {
        mesh.position.y = LAND_Y;
        this._spawnShockwave(wx, wz);
        // Bounce
        const bounceStart = performance.now();
        const bounce = (now2) => {
          if (this.disposed) return;
          const bt = Math.min((now2 - bounceStart) / BOUNCE_MS, 1);
          mesh.position.y = LAND_Y + Math.sin(bt * Math.PI) * BOUNCE_HEIGHT * (1 - bt * 0.4);
          if (bt < 1) {
            requestAnimationFrame(bounce);
          } else {
            mesh.position.y = LAND_Y;
            if (onLanded) onLanded();
          }
        };
        requestAnimationFrame(bounce);
      }
    };
    requestAnimationFrame(animate);
  }

  _schedulePortfolioRise(tiles, coreCx, coreCz) {
    const RISE_DEPTH = -10;
    const RIPPLE_SPEED = 0.22; // s per world unit from core
    const RISE_DURATION = 0.55;

    for (const tile of tiles) {
      const dist = Math.hypot((tile.wx || 0) - coreCx, (tile.wz || 0) - coreCz);
      const jitter = (this._hashCoord(
        Math.round((tile.wx || 0) * 10 + 1),
        Math.round((tile.wz || 0) * 10 + 1)
      ) - 0.5) * 0.08;
      const delay = Math.max(0, dist * RIPPLE_SPEED + jitter);

      const { mesh, finalY } = tile;
      const startPos = new THREE.Vector3(mesh.position.x, RISE_DEPTH, mesh.position.z);
      const endPos = new THREE.Vector3(mesh.position.x, finalY ?? 0.08, mesh.position.z);

      setTimeout(() => {
        if (this.disposed) return;
        this.tweens.push({ elapsed: 0, duration: RISE_DURATION, startPos, endPos, mesh, resolve: () => {} });
      }, delay * 1000);
    }
  }

  _spawnShockwave(wx, wz) {
    const DURATION = 600; // ms
    const scene = this.scene;

    const makeRing = (innerR, outerR, color, opacity) => {
      const mat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false
      });
      const geo = new THREE.RingGeometry(innerR, outerR, 32);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(wx, 0.05, wz);
      scene.add(mesh);
      return { mesh, mat };
    };

    const inner = makeRing(0.1, 0.35, 0xffffff, 0.75);
    const outer = makeRing(0.05, 0.18, 0xd4a832, 0.55);

    const start = performance.now();
    const expand = (now) => {
      if (this.disposed) { scene.remove(inner.mesh); scene.remove(outer.mesh); return; }
      const t = Math.min((now - start) / DURATION, 1);
      const ease = 1 - (1 - t) * (1 - t); // ease-out quad

      const iScale = 1 + ease * 5.5;
      inner.mesh.scale.set(iScale, iScale, 1);
      inner.mat.opacity = 0.75 * (1 - ease);

      const oScale = 1 + ease * 9.0;
      outer.mesh.scale.set(oScale, oScale, 1);
      outer.mat.opacity = 0.55 * (1 - ease * 0.9);

      if (t < 1) {
        requestAnimationFrame(expand);
      } else {
        scene.remove(inner.mesh);
        scene.remove(outer.mesh);
        inner.geo?.dispose(); outer.geo?.dispose();
        inner.mat.dispose(); outer.mat.dispose();
      }
    };
    requestAnimationFrame(expand);
  }

  async addBorrower(borrowerId, assetLoader, borrowerInfo, spawnIndex, spawnTotal, spawnPos) {
    if (this.borrowerMeshes.has(borrowerId)) {
      return this.borrowerMeshes.get(borrowerId);
    }
    const charPack = assetLoader?.getCharacterPack?.();
    const chars = Array.isArray(charPack?.characters) ? charPack.characters : [];
    const charFile = chars.length ? chars[Math.abs(hashStr(borrowerId)) % chars.length] : null;
    const charPackName = assetLoader?.getCharacterPackName?.() || 'blocky-characters';

    let model = null;
    let animations = null;
    if (!this.characterAssetsUnavailable && assetLoader?.manifest && charFile) {
      try {
        const loaded = await assetLoader.loadModel(charPackName, charFile);
        model = loaded.scene;
        animations = loaded.animations;
        model.scale.setScalar(0.35);
      } catch (e) {
        this.characterAssetsUnavailable = true;
        console.warn('Character assets unavailable, using fallback borrowers:', e);
      }
    }

    if (!model) {
      model = this._createFallbackBorrower();
    }

    // Spawn at explicit position, or fall back to residential road spread
    if (spawnPos) {
      const spread = (spawnIndex - (spawnTotal - 1) / 2) * 0.6;
      model.position.set(spawnPos.x + spread, 0, spawnPos.z);
    } else {
      const totalLength = this.layout?.totalLength || 10;
      const spacing = totalLength / (spawnTotal + 1);
      model.position.set(0, 0, spacing * (spawnIndex + 1));
    }

    model.userData.borrowerId = borrowerId;
    model.userData.borrowerName = borrowerInfo?.name || borrowerId;
    this.borrowerMeshes.set(borrowerId, model);
    this.borrowerStates.set(borrowerId, 'normal');
    this.scene.add(model);

    // Borrower label — placed above character head
    if (borrowerInfo) {
      const labelDiv = document.createElement('div');
      labelDiv.className = 'borrower-label';
      const name = borrowerInfo.name || borrowerId;
      const amount = borrowerInfo.amount ? `$${Math.round(borrowerInfo.amount).toLocaleString('en-US')}` : '';
      labelDiv.textContent = amount ? `${name}\n${amount}` : name;
      const label = new CSS2DObject(labelDiv);
      const labelY = this._borrowerLabelHeight(model);
      label.position.set(0, labelY, 0);
      model.add(label);
      this.borrowerLabels.set(borrowerId, label);
    }

    if (borrowerInfo?.state) {
      this.setBorrowerState(borrowerId, borrowerInfo.state);
    }

    // Try to play idle animation
    if (animations && animations.length > 0) {
      const mixer = new THREE.AnimationMixer(model);
      mixer.clipAction(animations[0]).play();
      this.mixers.push(mixer);
    }

    return model;
  }

  setBorrowerLabelVisible(borrowerId, visible) {
    const label = this.borrowerLabels.get(borrowerId);
    if (!label) return;
    label.element.style.display = visible ? '' : 'none';
  }

  setBorrowerState(borrowerId, state) {
    const mesh = this.borrowerMeshes.get(borrowerId);
    if (!mesh) return;
    this.borrowerStates.set(borrowerId, state);

    let bodyColor = 0x4a90d9;
    let emissive = 0x000000;
    if (state === 'defaulted' || state === 'bankrupt') {
      bodyColor = 0x3b3f4a;
      emissive = 0x090909;
    } else if (state === 'fraud') {
      bodyColor = 0x151515;
      emissive = 0x1a0000;
    } else if (state === 'repaid') {
      bodyColor = 0x4a7a4a;
      emissive = 0x001200;
    }

    mesh.traverse(c => {
      if (!c.isMesh || !c.material) return;
      const mats = Array.isArray(c.material) ? c.material : [c.material];
      for (const mat of mats) {
        if (mat.color) mat.color.setHex(bodyColor);
        if (mat.emissive) mat.emissive.setHex(emissive);
      }
    });

    const label = this.borrowerLabels.get(borrowerId);
    if (label) {
      label.element.classList.remove('borrower-defaulted', 'borrower-fraud', 'borrower-repaid');
      if (state === 'defaulted' || state === 'bankrupt') label.element.classList.add('borrower-defaulted');
      if (state === 'fraud') label.element.classList.add('borrower-fraud');
      if (state === 'repaid') label.element.classList.add('borrower-repaid');
    }
  }

  animateBorrowerWalk(borrowerId, target, duration = 1.0) {
    const mesh = this.borrowerMeshes.get(borrowerId);
    if (!mesh) return Promise.resolve();

    let endPos;
    if (typeof target === 'number') {
      // Legacy: target is a building index
      const bld = this.layout?.buildings?.[target];
      if (!bld) return Promise.resolve();
      endPos = new THREE.Vector3(bld.x * 0.6, 0, bld.z);
    } else if (target && typeof target === 'object') {
      // New: target is {x, z}
      endPos = new THREE.Vector3(target.x, 0, target.z);
    } else {
      return Promise.resolve();
    }

    const startPos = mesh.position.clone();

    // Face direction of travel
    mesh.lookAt(endPos);

    return new Promise(resolve => {
      const tween = { elapsed: 0, duration, startPos, endPos, mesh, resolve };
      this.tweens.push(tween);
    });
  }

  pulseBuildings(durationMs) {
    for (const [idx] of this.buildingMeshes) {
      this._setBuildingEmissive(idx, 0x1a3a6a);
    }
    return new Promise(resolve => {
      setTimeout(() => {
        for (const [idx] of this.buildingMeshes) {
          if (this.selectedBuilding !== idx) {
            this._setBuildingEmissive(idx, 0x000000);
          }
        }
        resolve();
      }, durationMs);
    });
  }

  fadeBorrower(borrowerId, durationMs = 500) {
    const mesh = this.borrowerMeshes.get(borrowerId);
    const label = this.borrowerLabels.get(borrowerId);
    if (!mesh) return Promise.resolve();

    // Collect all materials and make them transparent
    const materials = [];
    mesh.traverse(c => {
      if (c.isMesh && c.material) {
        const mat = c.material;
        mat.transparent = true;
        materials.push(mat);
      }
    });

    const labelEl = label?.element;
    const startTime = performance.now();

    return new Promise(resolve => {
      const tick = () => {
        const elapsed = performance.now() - startTime;
        const t = Math.min(elapsed / durationMs, 1);
        const opacity = 1 - t;

        for (const mat of materials) {
          mat.opacity = opacity;
        }
        if (labelEl) {
          labelEl.style.opacity = opacity;
        }

        if (t < 1) {
          requestAnimationFrame(tick);
        } else {
          // Clean up
          if (label) {
            mesh.remove(label);
            label.element?.remove();
          }
          this.scene.remove(mesh);
          this.borrowerMeshes.delete(borrowerId);
          this.borrowerLabels.delete(borrowerId);
          this.borrowerStates.delete(borrowerId);
          resolve();
        }
      };
      requestAnimationFrame(tick);
    });
  }

  highlightBuilding(lenderIndex, color = 0xe8a838) {
    // Reset previous
    if (this.selectedBuilding !== null) {
      this._setBuildingEmissive(this.selectedBuilding, 0x000000);
    }
    this.selectedBuilding = lenderIndex;
    if (lenderIndex !== null) {
      this._setBuildingEmissive(lenderIndex, color);
    }
  }

  showDecision(lenderIndex, approved) {
    const color = approved ? 0x4ae84a : 0xe84a4a;
    this._setBuildingEmissive(lenderIndex, color);
    setTimeout(() => {
      if (this.selectedBuilding !== lenderIndex) {
        this._setBuildingEmissive(lenderIndex, 0x000000);
      }
    }, 1500);
  }

  clearBorrowers() {
    for (const [borrowerId, mesh] of this.borrowerMeshes) {
      const label = this.borrowerLabels.get(borrowerId);
      if (label) {
        mesh.remove(label);
        label.element?.remove();
      }
      this.scene.remove(mesh);
    }
    this.borrowerMeshes.clear();
    this.borrowerLabels.clear();
    this.borrowerStates.clear();
    this.mixers = [];
  }

  resetBuildings() {
    for (const [idx] of this.buildingMeshes) {
      this._setBuildingEmissive(idx, 0x000000);
      const slots = this.portfolioHexMeshes.get(idx) || [];
      for (const hex of slots) {
        const mats = Array.isArray(hex.material) ? hex.material : [hex.material];
        for (const mat of mats) {
          if (mat?.emissive) mat.emissive.setHex(0x000000);
          if (mat?.color && mat.userData?._baseColorHex !== undefined) {
            mat.color.setHex(mat.userData._baseColorHex);
          }
        }
      }
    }
    this.selectedBuilding = null;
  }

  updateLabel(lenderIndex, text) {
    // Legacy fallback — updates the signpost name only
    const label = this.buildingLabels.get(lenderIndex);
    if (label) {
      const nameEl = label.element.querySelector('.signpost-name');
      if (nameEl) nameEl.textContent = text;
      else label.element.textContent = text;
    }
  }

  updateSignpost(lenderIndex, name, approvalRate, pnl, avgLoanSize = null, aum = null) {
    const label = this.buildingLabels.get(lenderIndex);
    if (!label) return;
    const el = label.element;

    const nameEl = el.querySelector('.signpost-name');
    if (nameEl) nameEl.textContent = name;

    // Update collapsed summary line
    const summaryEl = el.querySelector('.signpost-summary');
    if (summaryEl) {
      if (pnl !== null && pnl !== undefined) {
        const pnlStr = pnl >= 0 ? `+$${fmtK(pnl)}` : `-$${fmtK(Math.abs(pnl))}`;
        summaryEl.textContent = pnlStr;
        summaryEl.className = 'signpost-summary ' + (pnl >= 0 ? 'positive' : 'negative');
      } else {
        summaryEl.textContent = '';
        summaryEl.className = 'signpost-summary';
      }
    }

    // Update expanded grid cells
    const cells = el.querySelectorAll('.signpost-cell-value');
    if (cells.length >= 4) {
      // Approve %
      cells[0].textContent = approvalRate !== null ? `${Math.round(approvalRate)}%` : '—';

      // P&L
      if (pnl === null || pnl === undefined) {
        cells[1].textContent = '—';
        cells[1].className = 'signpost-cell-value';
      } else {
        const pnlStr = pnl >= 0 ? `+$${fmtK(pnl)}` : `-$${fmtK(Math.abs(pnl))}`;
        cells[1].textContent = pnlStr;
        cells[1].className = 'signpost-cell-value ' + (pnl >= 0 ? 'positive' : 'negative');
      }

      // Avg Loan
      cells[2].textContent = avgLoanSize !== null && avgLoanSize !== undefined ? `$${fmtK(avgLoanSize)}` : '—';

      // AUM
      cells[3].textContent = aum !== null && aum !== undefined ? `$${fmtK(aum)}` : '—';
    }
  }

  updateLenderBorrowerStack(lenderIndex, borrowerNames = []) {
    const label = this.buildingLabels.get(lenderIndex);
    if (!label) return;
    const stack = label.element.querySelector('.signpost-borrowers');
    if (!stack) return;
    if (!borrowerNames.length) {
      stack.innerHTML = '';
      return;
    }
    stack.innerHTML = borrowerNames
      .slice(0, 8)
      .map(name => `<div class="signpost-borrower">${escapeHtml(name)}</div>`)
      .join('');
  }

  _signpostHTML(name, slotCount = null) {
    let html =
      `<div class="signpost-name">${escapeHtml(name)}</div>` +
      `<div class="signpost-summary"></div>` +
      `<div class="signpost-detail">` +
        `<div class="signpost-grid">` +
          `<div class="signpost-cell"><div class="signpost-cell-label">Approve</div><div class="signpost-cell-value">—</div></div>` +
          `<div class="signpost-cell"><div class="signpost-cell-label">P&amp;L</div><div class="signpost-cell-value">—</div></div>` +
          `<div class="signpost-cell"><div class="signpost-cell-label">Avg Loan</div><div class="signpost-cell-value">—</div></div>` +
          `<div class="signpost-cell"><div class="signpost-cell-label">AUM</div><div class="signpost-cell-value">—</div></div>` +
        `</div>` +
        `<div class="signpost-borrowers"></div>`;
    if (slotCount !== null) {
      html += `<div class="portfolio-capacity">${slotCount} slots</div>`;
    }
    html += `</div>`;
    return html;
  }

  _toggleSignpost(labelDiv) {
    // Close any other expanded signpost
    this._closeAllSignposts(labelDiv);
    labelDiv.classList.toggle('expanded');

    // Add/remove click-outside listener
    if (labelDiv.classList.contains('expanded')) {
      this._outsideClickHandler = (e) => {
        if (!labelDiv.contains(e.target)) {
          labelDiv.classList.remove('expanded');
          document.removeEventListener('pointerdown', this._outsideClickHandler, true);
          this._outsideClickHandler = null;
        }
      };
      // Use capture + delay so the current click doesn't immediately close it
      setTimeout(() => {
        if (this._outsideClickHandler) {
          document.addEventListener('pointerdown', this._outsideClickHandler, true);
        }
      }, 0);
    } else if (this._outsideClickHandler) {
      document.removeEventListener('pointerdown', this._outsideClickHandler, true);
      this._outsideClickHandler = null;
    }
  }

  _closeAllSignposts(except = null) {
    const root = this.labelRenderer?.domElement || this.container;
    const allSignposts = root?.querySelectorAll('.bank-signpost.expanded') || [];
    for (const sp of allSignposts) {
      if (sp !== except) sp.classList.remove('expanded');
    }
    if (this._outsideClickHandler) {
      document.removeEventListener('pointerdown', this._outsideClickHandler, true);
      this._outsideClickHandler = null;
    }
  }

  updateLenderVisualState(lenderIndex, state = {}) {
    const badge = this.buildingStatusBadges.get(lenderIndex);
    const smoke = this.buildingSmokeBadges.get(lenderIndex);
    const label = this.buildingLabels.get(lenderIndex);
    const group = this.buildingMeshes.get(lenderIndex);
    if (!group) return;

    if (badge) {
      badge.element.textContent = state.winner ? '★' : '';
      badge.element.className = 'lender-status-badge' + (state.winner ? ' winner' : '');
    }

    if (smoke) {
      smoke.element.textContent = state.highlyLeveraged ? '~~~' : '';
      smoke.element.className = 'lender-smoke' + (state.highlyLeveraged ? ' active' : '');
    }

    if (label) {
      label.element.classList.toggle('lender-bankrupt', !!state.bankrupt);
    }

    group.traverse(c => {
      if (!c.isMesh || !c.material) return;
      const mats = Array.isArray(c.material) ? c.material : [c.material];
      for (const mat of mats) {
        if (mat.color && mat.userData._baseColorHex === undefined) {
          mat.userData._baseColorHex = mat.color.getHex();
        }
        if (state.bankrupt) {
          if (mat.color) {
            const base = new THREE.Color(mat.userData._baseColorHex);
            base.multiplyScalar(0.45);
            mat.color.copy(base);
          }
          if (mat.emissive) mat.emissive.setHex(0x0a0a0a);
        } else {
          if (mat.color && mat.userData._baseColorHex !== undefined) {
            mat.color.setHex(mat.userData._baseColorHex);
          }
          if (mat.emissive) mat.emissive.setHex(0x000000);
        }
      }
    });

    const slots = this.portfolioHexMeshes.get(lenderIndex) || [];
    for (const hex of slots) {
      const mats = Array.isArray(hex.material) ? hex.material : [hex.material];
      for (const mat of mats) {
        if (!mat?.color) continue;
        if (mat.userData._baseColorHex === undefined) {
          mat.userData._baseColorHex = mat.color.getHex();
        }
        if (state.bankrupt) {
          const base = new THREE.Color(mat.userData._baseColorHex);
          base.multiplyScalar(0.35);
          mat.color.copy(base);
          if (mat.emissive) mat.emissive.setHex(0x050505);
        } else {
          mat.color.setHex(mat.userData._baseColorHex);
          if (mat.emissive) mat.emissive.setHex(state.winner ? 0x302000 : 0x000000);
        }
      }
    }
  }

  _createPortfolioCore(lenderIndex) {
    const group = new THREE.Group();
    const hue = (lenderIndex * 0.19) % 1;
    const tone = new THREE.Color().setHSL(hue, 0.45, 0.42);

    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(0.62, 0.72, 0.28, 6),
      new THREE.MeshStandardMaterial({
        color: tone.clone().multiplyScalar(0.68),
        roughness: 0.85,
        metalness: 0.12,
      })
    );
    base.rotation.y = Math.PI / 6;
    base.position.y = 0.14;
    base.castShadow = true;
    base.receiveShadow = true;
    group.add(base);

    const tower = new THREE.Mesh(
      new THREE.CylinderGeometry(0.34, 0.44, 1.4, 6),
      new THREE.MeshStandardMaterial({
        color: tone,
        roughness: 0.7,
        metalness: 0.18,
      })
    );
    tower.rotation.y = Math.PI / 6;
    tower.position.y = 0.98;
    tower.castShadow = true;
    tower.receiveShadow = true;
    group.add(tower);

    const cap = new THREE.Mesh(
      new THREE.CylinderGeometry(0.4, 0.4, 0.1, 6),
      new THREE.MeshStandardMaterial({
        color: 0xd9c38a,
        roughness: 0.6,
        metalness: 0.28,
      })
    );
    cap.rotation.y = Math.PI / 6;
    cap.position.y = 1.72;
    cap.castShadow = true;
    group.add(cap);

    return group;
  }

  _createPortfolioHexCell(lenderIndex, radius = 0.52) {
    const hue = (lenderIndex * 0.19) % 1;
    const baseColor = new THREE.Color().setHSL(hue, 0.38, 0.3);
    const mat = new THREE.MeshStandardMaterial({
      color: baseColor,
      roughness: 0.9,
      metalness: 0.1,
      emissive: 0x000000,
    });
    const hex = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 0.12, 6), mat);
    hex.rotation.y = Math.PI / 6;
    hex.castShadow = true;
    hex.receiveShadow = true;
    return hex;
  }

  _createFallbackRoad(road) {
    const group = new THREE.Group();
    const isJunction = /junction|cross|tsplit/.test(road.modelFile || '');
    const isBridge = !!road.isBridge;
    const width = isJunction ? 2.1 : 1.1;
    const length = isJunction ? 2.1 : 2.0;
    const asphalt = new THREE.Mesh(
      new THREE.BoxGeometry(width, 0.04, length),
      new THREE.MeshStandardMaterial({
        color: isBridge ? 0x4a515e : 0x2a3444,
        roughness: 0.95,
        metalness: isBridge ? 0.3 : 0.05,
      })
    );
    asphalt.position.y = isBridge ? 0.28 : 0.02;
    asphalt.receiveShadow = true;
    group.add(asphalt);

    if (!isJunction) {
      const line = new THREE.Mesh(
        new THREE.BoxGeometry(0.08, 0.01, length * 0.7),
        new THREE.MeshStandardMaterial({ color: isBridge ? 0xd3dae5 : 0xe8a838, roughness: 0.8 })
      );
      line.position.y = isBridge ? 0.31 : 0.05;
      group.add(line);
    }
    if (isBridge) {
      const railMat = new THREE.MeshStandardMaterial({ color: 0x8a95a6, roughness: 0.6, metalness: 0.45 });
      const railL = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.16, length), railMat);
      const railR = railL.clone();
      railL.position.set(-(width / 2) + 0.04, 0.36, 0);
      railR.position.set((width / 2) - 0.04, 0.36, 0);
      group.add(railL);
      group.add(railR);
    }
    return group;
  }

  _createFallbackBuilding(lenderIndex) {
    const group = new THREE.Group();
    const height = 1.7 + (lenderIndex % 4) * 0.35;
    const color = new THREE.Color().setHSL((lenderIndex * 0.17) % 1, 0.35, 0.42);

    const tower = new THREE.Mesh(
      new THREE.BoxGeometry(1.3, height, 1.3),
      new THREE.MeshStandardMaterial({ color, roughness: 0.8, metalness: 0.1 })
    );
    tower.position.y = height / 2;
    tower.castShadow = true;
    tower.receiveShadow = true;
    group.add(tower);

    const roof = new THREE.Mesh(
      new THREE.ConeGeometry(0.95, 0.6, 4),
      new THREE.MeshStandardMaterial({ color: 0xd9c38a, roughness: 0.85 })
    );
    roof.position.y = height + 0.3;
    roof.rotation.y = Math.PI / 4;
    roof.castShadow = true;
    group.add(roof);

    const door = new THREE.Mesh(
      new THREE.BoxGeometry(0.28, 0.6, 0.04),
      new THREE.MeshStandardMaterial({ color: 0x111a2a, roughness: 0.9 })
    );
    door.position.set(0, 0.3, 0.67);
    group.add(door);
    return group;
  }

  _createFallbackTownBuilding() {
    const group = new THREE.Group();
    const width = 1.1 + Math.random() * 0.4;
    const depth = 1.1 + Math.random() * 0.4;
    const height = 1.1 + Math.random() * 1.0;
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, depth),
      new THREE.MeshStandardMaterial({ color: 0x55627a, roughness: 0.9, metalness: 0.05 })
    );
    body.position.y = height / 2;
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);
    return group;
  }

  _createFallbackIndustrialBuilding() {
    const group = new THREE.Group();
    const shell = new THREE.Mesh(
      new THREE.BoxGeometry(1.9, 1.1, 1.5),
      new THREE.MeshStandardMaterial({ color: 0x4b5666, roughness: 0.92, metalness: 0.08 })
    );
    shell.position.y = 0.55;
    shell.castShadow = true;
    shell.receiveShadow = true;
    group.add(shell);

    const roof = new THREE.Mesh(
      new THREE.BoxGeometry(2.0, 0.08, 1.6),
      new THREE.MeshStandardMaterial({ color: 0x2f3948, roughness: 0.85, metalness: 0.2 })
    );
    roof.position.y = 1.12;
    roof.castShadow = true;
    group.add(roof);

    const stack = new THREE.Mesh(
      new THREE.CylinderGeometry(0.11, 0.11, 0.8, 12),
      new THREE.MeshStandardMaterial({ color: 0x7c8696, roughness: 0.7, metalness: 0.4 })
    );
    stack.position.set(0.45, 1.4, 0);
    stack.castShadow = true;
    group.add(stack);
    return group;
  }

  _createFallbackVehicle() {
    const group = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.9, 0.25, 1.5),
      new THREE.MeshStandardMaterial({ color: 0x5c7ea3, roughness: 0.75, metalness: 0.2 })
    );
    body.position.y = 0.2;
    body.castShadow = true;
    group.add(body);

    const cab = new THREE.Mesh(
      new THREE.BoxGeometry(0.75, 0.25, 0.7),
      new THREE.MeshStandardMaterial({ color: 0xa7b7cc, roughness: 0.7, metalness: 0.2 })
    );
    cab.position.set(0, 0.38, -0.15);
    cab.castShadow = true;
    group.add(cab);
    return group;
  }

  _createFallbackProp(propType) {
    const group = new THREE.Group();
    if (propType === 'streetlight') {
      const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.04, 0.05, 1.2, 8),
        new THREE.MeshStandardMaterial({ color: 0x9aa0b0, roughness: 0.6, metalness: 0.3 })
      );
      pole.position.y = 0.6;
      pole.castShadow = true;
      group.add(pole);

      const lamp = new THREE.Mesh(
        new THREE.SphereGeometry(0.08, 10, 8),
        new THREE.MeshStandardMaterial({ color: 0xe8a838, emissive: 0x2a1a00, emissiveIntensity: 0.8 })
      );
      lamp.position.y = 1.2;
      group.add(lamp);
      return group;
    }

    if (propType === 'bench') {
      const seat = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 0.08, 0.18),
        new THREE.MeshStandardMaterial({ color: 0x7a5a3c, roughness: 0.9 })
      );
      seat.position.y = 0.24;
      seat.castShadow = true;
      group.add(seat);
      return group;
    }

    const bush = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 10, 8),
      new THREE.MeshStandardMaterial({ color: 0x3f6b3d, roughness: 0.95 })
    );
    bush.position.y = 0.18;
    bush.castShadow = true;
    group.add(bush);
    return group;
  }

  _createFallbackBorrower() {
    const group = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.16, 0.2, 0.75, 10),
      new THREE.MeshStandardMaterial({ color: 0x4a90d9, roughness: 0.85, metalness: 0.05 })
    );
    body.position.y = 0.38;
    body.castShadow = true;
    group.add(body);

    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 12, 10),
      new THREE.MeshStandardMaterial({ color: 0xe8c8a6, roughness: 0.9 })
    );
    head.position.y = 0.9;
    head.castShadow = true;
    group.add(head);
    return group;
  }

  _borrowerLabelHeight(model) {
    try {
      const box = new THREE.Box3().setFromObject(model);
      const h = box.max.y - box.min.y;
      if (!Number.isFinite(h) || h <= 0) return 1.2;
      return Math.max(1.05, h + 0.08);
    } catch {
      return 1.2;
    }
  }

  dispose() {
    this.disposed = true;
    this.tweens = [];
    this.mixers = [];
    if (this._outsideClickHandler) {
      document.removeEventListener('pointerdown', this._outsideClickHandler, true);
      this._outsideClickHandler = null;
    }
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    if (this.renderer?.domElement && this._boundPointerMove) {
      this.renderer.domElement.removeEventListener('pointermove', this._boundPointerMove);
    }
    if (this.renderer?.domElement && this._boundClick) {
      this.renderer.domElement.removeEventListener('click', this._boundClick);
    }
    if (this._boundResize) {
      window.removeEventListener('resize', this._boundResize);
    }
    this.clearBorrowers();
    if (this.renderer) {
      this.renderer.dispose();
      if (this.container?.contains(this.renderer.domElement)) {
        this.container.removeChild(this.renderer.domElement);
      }
    }
    if (this.labelRenderer) {
      if (this.container?.contains(this.labelRenderer.domElement)) {
        this.container.removeChild(this.labelRenderer.domElement);
      }
    }
    this.controls?.dispose?.();
    this.renderer = null;
    this.labelRenderer = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
  }

  // ---- Private ----

  _animate() {
    if (this.disposed || !this.renderer || !this.scene || !this.camera) return;
    this.animationFrame = requestAnimationFrame(this._animate);
    const delta = this.clock.getDelta();

    this.controls?.update();

    // Update animation mixers
    for (const mixer of this.mixers) {
      mixer.update(delta);
    }

    // Update tweens
    for (let i = this.tweens.length - 1; i >= 0; i--) {
      const tw = this.tweens[i];
      tw.elapsed += delta;
      const t = Math.min(tw.elapsed / tw.duration, 1);
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      tw.mesh.position.lerpVectors(tw.startPos, tw.endPos, eased);
      if (t >= 1) {
        tw.resolve();
        this.tweens.splice(i, 1);
      }
    }

    this.renderer.render(this.scene, this.camera);
    this.labelRenderer?.render(this.scene, this.camera);
  }

  _onResize() {
    if (!this.container || !this.camera || !this.renderer || !this.labelRenderer) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
  }

  _onPointerMove(e) {
    if (!this.renderer || !this.camera) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hit = this._pickBuilding();
    const idx = hit?.object?.userData?.lenderIndex ??
      this._findParentLenderIndex(hit?.object);

    if (idx !== this.hoveredBuilding) {
      if (this.hoveredBuilding !== null && this.hoveredBuilding !== this.selectedBuilding) {
        this._setBuildingEmissive(this.hoveredBuilding, 0x000000);
      }
      this.hoveredBuilding = idx;
      if (idx !== null && idx !== this.selectedBuilding) {
        this._setBuildingEmissive(idx, 0x333333);
      }
      this.renderer.domElement.style.cursor = idx !== null ? 'pointer' : '';
      if (this.onBuildingHover) this.onBuildingHover(idx);
    }
  }

  _onClick(e) {
    if (!this.renderer || !this.camera) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hit = this._pickBuilding();
    const idx = hit?.object?.userData?.lenderIndex ??
      this._findParentLenderIndex(hit?.object);
    if (idx !== null && idx !== undefined) {
      this.highlightBuilding(idx);
      if (this.onBuildingClick) this.onBuildingClick(idx);
    }
  }

  _pickBuilding() {
    const meshes = [];
    for (const [, group] of this.buildingMeshes) {
      group.traverse(c => { if (c.isMesh) meshes.push(c); });
    }
    const hits = this.raycaster.intersectObjects(meshes, false);
    return hits[0] || null;
  }

  _findParentLenderIndex(obj) {
    while (obj) {
      if (obj.userData?.lenderIndex !== undefined) return obj.userData.lenderIndex;
      obj = obj.parent;
    }
    return null;
  }

  _setBuildingEmissive(lenderIndex, color) {
    const group = this.buildingMeshes.get(lenderIndex);
    if (!group) return;
    group.traverse(c => {
      if (c.isMesh && c.material) {
        const mats = Array.isArray(c.material) ? c.material : [c.material];
        for (const mat of mats) {
          if (mat.emissive) mat.emissive.setHex(color);
        }
      }
    });
  }
}

function hashStr(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  }
  return h;
}

function fmtK(n) {
  const abs = Math.abs(n);
  if (abs >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (n / 1e3).toFixed(0) + 'K';
  return Math.round(n).toString();
}

function escapeHtml(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}
