// texture-loader.js
export class TextureLoader {
    constructor(device) {
        this.device = device;
        this.textureCache = new Map();
    }

    static async initialize(device) {
        const instance = new TextureLoader(device);
        return instance;
    }

    async loadImage(url) {
        if (this.textureCache.has(url)) {
            return this.textureCache.get(url);
        }

        const response = await fetch(url);
        const blob = await response.blob();
        const imageBitmap = await createImageBitmap(blob);
        const texture = await this.createTextureFromBitmap(imageBitmap);
        
        this.textureCache.set(url, texture);
        return texture;
    }

    async loadImageFromFile(file) {
        const imageBitmap = await createImageBitmap(file);
        return this.createTextureFromBitmap(imageBitmap);
    }

    async createTextureFromBitmap(bitmap) {
        const texture = this.device.createTexture({
            size: [bitmap.width, bitmap.height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING |
                   GPUTextureUsage.COPY_DST |
                   GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: texture },
            [bitmap.width, bitmap.height]
        );

        const sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });

        return {
            texture,
            sampler,
            width: bitmap.width,
            height: bitmap.height
        };
    }

    releaseTexture(url) {
        const texture = this.textureCache.get(url);
        if (texture) {
            texture.texture.destroy();
            this.textureCache.delete(url);
        }
    }

    clearCache() {
        for (const texture of this.textureCache.values()) {
            texture.texture.destroy();
        }
        this.textureCache.clear();
    }
}