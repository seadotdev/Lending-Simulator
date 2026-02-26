import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

export class TownScene {
  constructor() {
    this.renderer = null;
    this.labelRenderer = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
    this.clock = new THREE.Clock();

    this.buildingMeshes = new Map();  // lenderIndex → Group
    this.buildingLabels = new Map();  // lenderIndex → CSS2DObject
    this.borrowerMeshes = new Map();  // borrowerId → Group
    this.borrowerLabels = new Map();  // borrowerId → CSS2DObject
    this.mixers = [];                 // AnimationMixer[]
    this.tweens = [];                 // active tweens
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.hoveredBuilding = null;
    this.selectedBuilding = null;
    this.characterAssetsUnavailable = false;

    this.onBuildingClick = null;      // callback(lenderIndex)
    this.onBuildingHover = null;      // callback(lenderIndex|null)

    this._animate = this._animate.bind(this);
  }

  init(container) {
    this.container = container;
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
    this.scene.background = new THREE.Color(0x0f1520);
    this.scene.fog = new THREE.Fog(0x0f1520, 30, 70);

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
    const ambient = new THREE.AmbientLight(0x404060, 0.6);
    this.scene.add(ambient);

    const hemi = new THREE.HemisphereLight(0x88aacc, 0x443322, 0.5);
    this.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffeedd, 1.2);
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
    ctx.fillStyle = '#1a3320';
    ctx.fillRect(0, 0, 512, 512);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    const gridStep = 32;
    for (let x = 0; x <= 512; x += gridStep) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 512); ctx.stroke();
    }
    for (let y = 0; y <= 512; y += gridStep) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(512, y); ctx.stroke();
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
    this.renderer.domElement.addEventListener('pointermove', (e) => this._onPointerMove(e));
    this.renderer.domElement.addEventListener('click', (e) => this._onClick(e));
    window.addEventListener('resize', () => this._onResize());

    this._animate();
  }

  async buildTown(layout, lenderNames, assetLoader) {
    this.layout = layout;
    const activePack = assetLoader.manifest.activePack;
    const packConfig = assetLoader.getActivePack();
    this.characterAssetsUnavailable = false;
    let packAssetsUnavailable = false;
    let fallbackUsed = false;

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
      model.position.set(road.x, 0, road.z);
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
      labelDiv.innerHTML =
        `<div class="signpost-name">${name}</div>` +
        `<div class="signpost-stats">` +
        `<div class="signpost-stat"><div class="signpost-stat-label">Approve</div><div class="signpost-stat-value">—</div></div>` +
        `<div class="signpost-stat"><div class="signpost-stat-label">P&L</div><div class="signpost-stat-value">—</div></div>` +
        `</div>`;
      const label = new CSS2DObject(labelDiv);
      label.position.set(0, 2.5, 0);
      model.add(label);
      this.buildingLabels.set(bld.id, label);
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

  async addBorrower(borrowerId, assetLoader, borrowerInfo, spawnIndex, spawnTotal, spawnPos) {
    if (!assetLoader?.manifest) return;
    const charPack = assetLoader.getCharacterPack();
    if (!charPack?.characters) return;
    const chars = charPack.characters;
    const charFile = chars[Math.abs(hashStr(borrowerId)) % chars.length];
    const charPackName = assetLoader.getCharacterPackName();

    let model = null;
    let animations = null;
    if (!this.characterAssetsUnavailable) {
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
    this.borrowerMeshes.set(borrowerId, model);
    this.scene.add(model);

    // Borrower label — placed above character head
    if (borrowerInfo) {
      const labelDiv = document.createElement('div');
      labelDiv.className = 'borrower-label';
      const name = borrowerInfo.name || borrowerId;
      const amount = borrowerInfo.amount ? `$${Math.round(borrowerInfo.amount).toLocaleString('en-US')}` : '';
      labelDiv.textContent = amount ? `${name}\n${amount}` : name;
      const label = new CSS2DObject(labelDiv);
      label.position.set(0, 3.5, 0);
      model.add(label);
      this.borrowerLabels.set(borrowerId, label);
    }

    // Try to play idle animation
    if (animations && animations.length > 0) {
      const mixer = new THREE.AnimationMixer(model);
      mixer.clipAction(animations[0]).play();
      this.mixers.push(mixer);
    }

    return model;
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
          this.scene.remove(mesh);
          this.borrowerMeshes.delete(borrowerId);
          this.borrowerLabels.delete(borrowerId);
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
    for (const [, mesh] of this.borrowerMeshes) {
      this.scene.remove(mesh);
    }
    this.borrowerMeshes.clear();
    this.borrowerLabels.clear();
    this.mixers = [];
  }

  resetBuildings() {
    for (const [idx] of this.buildingMeshes) {
      this._setBuildingEmissive(idx, 0x000000);
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

  updateSignpost(lenderIndex, name, approvalRate, pnl) {
    const label = this.buildingLabels.get(lenderIndex);
    if (!label) return;
    const el = label.element;

    const nameEl = el.querySelector('.signpost-name');
    if (nameEl) nameEl.textContent = name;

    const values = el.querySelectorAll('.signpost-stat-value');
    if (values.length >= 2) {
      // Approval rate
      values[0].textContent = approvalRate !== null ? `${Math.round(approvalRate)}%` : '—';

      // P&L
      const pnlStr = pnl >= 0 ? `+$${fmtK(pnl)}` : `-$${fmtK(Math.abs(pnl))}`;
      values[1].textContent = pnl !== null ? pnlStr : '—';
      values[1].className = 'signpost-stat-value ' + (pnl >= 0 ? 'positive' : 'negative');
    }
  }

  _createFallbackRoad(road) {
    const group = new THREE.Group();
    const isJunction = /junction|cross|tsplit/.test(road.modelFile || '');
    const width = isJunction ? 2.1 : 1.1;
    const length = isJunction ? 2.1 : 2.0;
    const asphalt = new THREE.Mesh(
      new THREE.BoxGeometry(width, 0.04, length),
      new THREE.MeshStandardMaterial({ color: 0x2a3444, roughness: 0.95, metalness: 0.05 })
    );
    asphalt.position.y = 0.02;
    asphalt.receiveShadow = true;
    group.add(asphalt);

    if (!isJunction) {
      const line = new THREE.Mesh(
        new THREE.BoxGeometry(0.08, 0.01, length * 0.7),
        new THREE.MeshStandardMaterial({ color: 0xe8a838, roughness: 0.8 })
      );
      line.position.y = 0.05;
      group.add(line);
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

  dispose() {
    this.tweens = [];
    this.mixers = [];
    if (this.renderer) {
      this.renderer.dispose();
      this.container.removeChild(this.renderer.domElement);
    }
    if (this.labelRenderer) {
      this.container.removeChild(this.labelRenderer.domElement);
    }
  }

  // ---- Private ----

  _animate() {
    requestAnimationFrame(this._animate);
    const delta = this.clock.getDelta();

    this.controls.update();

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
    this.labelRenderer.render(this.scene, this.camera);
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
  }

  _onPointerMove(e) {
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

  _onClick() {
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
        const mat = c.material;
        if (mat.emissive) mat.emissive.setHex(color);
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
