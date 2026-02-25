import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export class AssetLoader {
  constructor() {
    this.manifest = null;
    this.loader = new GLTFLoader();
    this.cache = new Map();        // key → Promise<GLTF>
    this.animations = new Map();   // key → AnimationClip[]
  }

  async loadManifest(url = 'asset-manifest.json') {
    const resp = await fetch(url);
    this.manifest = await resp.json();
    return this.manifest;
  }

  getActivePack() {
    return this.manifest?.packs[this.manifest.activePack];
  }

  getCharacterPack() {
    const name = this.manifest?.characterPack || 'adventurers';
    return this.manifest?.packs[name];
  }

  getCharacterPackName() {
    return this.manifest?.characterPack || 'adventurers';
  }

  getPackConfig(packName) {
    return this.manifest?.packs[packName];
  }

  async loadModel(packName, fileName) {
    const pack = this.manifest.packs[packName];
    if (!pack) throw new Error(`Unknown pack: ${packName}`);
    const url = `${pack.basePath}/${fileName}`;
    if (!this.cache.has(url)) {
      this.cache.set(url, new Promise((resolve, reject) => {
        this.loader.load(url, resolve, undefined, reject);
      }));
    }
    const gltf = await this.cache.get(url);
    const clone = gltf.scene.clone(true);
    return { scene: clone, animations: gltf.animations };
  }

  async loadAnimations(packName, fileName) {
    const key = `${packName}/${fileName}`;
    if (!this.animations.has(key)) {
      const pack = this.manifest.packs[packName];
      const url = `${pack.basePath}/${fileName}`;
      const promise = new Promise((resolve, reject) => {
        this.loader.load(url, (gltf) => resolve(gltf.animations), undefined, reject);
      });
      this.animations.set(key, promise);
    }
    return this.animations.get(key);
  }

  async preloadBuildings() {
    const pack = this.getActivePack();
    if (!pack?.buildings) return;
    const promises = pack.buildings.map(f =>
      this.loadModel(this.manifest.activePack, f)
    );
    await Promise.all(promises);
  }
}
