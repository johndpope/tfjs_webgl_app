// mask-manager.js
export class MaskManager {
    constructor(device, textureLoader) {
        if (!device) throw new Error('GPU device is required');
        if (!textureLoader) throw new Error('TextureLoader is required');

        this.device = device;
        this.textureLoader = textureLoader;
        this.masks = new Map();
        this.activeMaskId = null;
        this.maskQueue = [];
        this.maxMasks = 5;
        
        // Create bind group layout during initialization
        this.bindGroupLayout = this.createBindGroupLayout();
    }

    createBindGroupLayout() {
        try {
            return this.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.FRAGMENT,
                        sampler: { type: 'filtering' }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: { sampleType: 'float' }
                    }
                ]
            });
        } catch (error) {
            console.error('Failed to create bind group layout:', error);
            throw new Error('Failed to initialize MaskManager: ' + error.message);
        }
    }

    async addMask(id, imageUrlOrFile) {
        if (!id) throw new Error('Mask ID is required');
        if (!imageUrlOrFile) throw new Error('Image source is required');

        try {
            // Load texture
            const texture = typeof imageUrlOrFile === 'string'
                ? await this.textureLoader.loadImage(imageUrlOrFile)
                : await this.textureLoader.loadImageFromFile(imageUrlOrFile);

            if (!texture) throw new Error('Failed to load texture');

            // Create mask metadata
            const mask = {
                id,
                texture,
                landmarks: null,
                bindGroup: null,
                lastUsed: Date.now()
            };

            // Create bind group
            try {
                mask.bindGroup = this.createBindGroup(mask);
            } catch (bindError) {
                console.error('Failed to create bind group:', bindError);
                // Cleanup texture if bind group creation fails
                texture.texture?.destroy?.();
                throw bindError;
            }

            // Add to collections
            this.masks.set(id, mask);
            this.maskQueue.push(id);

            // Set as active if first mask
            if (!this.activeMaskId) {
                this.activeMaskId = id;
            }

            // Cleanup old masks
            this.cleanupOldMasks();

            return mask;
        } catch (error) {
            console.error(`Failed to add mask ${id}:`, error);
            throw error;
        }
    }

    createBindGroup(mask) {
        if (!this.bindGroupLayout) {
            throw new Error('Bind group layout not initialized');
        }

        if (!mask?.texture?.texture || !mask?.texture?.sampler) {
            throw new Error('Invalid mask texture or sampler');
        }

        try {
            return this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: mask.texture.sampler
                    },
                    {
                        binding: 1,
                        resource: mask.texture.texture.createView()
                    }
                ]
            });
        } catch (error) {
            console.error('Failed to create bind group:', error);
            throw error;
        }
    }

    getActiveMask() {
        const mask = this.masks.get(this.activeMaskId);
        if (!mask) {
            console.warn('No active mask found');
            return null;
        }
        return mask;
    }

    async setActiveMask(id) {
        if (!this.masks.has(id)) {
            throw new Error(`Mask with id ${id} not found`);
        }

        try {
            const mask = this.masks.get(id);
            mask.lastUsed = Date.now();
            this.activeMaskId = id;

            // Compute landmarks if needed
            if (!mask.landmarks) {
                mask.landmarks = await this.computeFaceLandmarks(mask.texture);
            }

            return mask;
        } catch (error) {
            console.error(`Failed to set active mask ${id}:`, error);
            throw error;
        }
    }

    async computeFaceLandmarks(texture) {
        if (!texture || !texture.texture) {
            throw new Error('Invalid texture for landmark computation');
        }

        try {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            if (!ctx) {
                throw new Error('Failed to get 2D context');
            }

            canvas.width = texture.width;
            canvas.height = texture.height;
            
            // Draw texture to canvas
            ctx.drawImage(texture.texture, 0, 0);
            
            // Load face detection model
            const model = await faceLandmarksDetection.createDetector(
                faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
                {
                    runtime: 'tfjs',
                    refineLandmarks: true,
                    maxFaces: 1,
                }
            );
            
            const predictions = await model.estimateFaces(canvas);
            return predictions.length > 0 ? predictions[0].keypoints : null;

        } catch (error) {
            console.error('Failed to compute face landmarks:', error);
            return null;
        }
    }

    cleanupOldMasks() {
        while (this.maskQueue.length > this.maxMasks) {
            const oldestMaskId = this.maskQueue.shift();
            if (oldestMaskId !== this.activeMaskId) {
                this.removeMask(oldestMaskId);
            }
        }
    }

    removeMask(id) {
        try {
            const mask = this.masks.get(id);
            if (mask) {
                // Cleanup GPU resources
                mask.texture.texture?.destroy?.();
                this.masks.delete(id);
                
                // Update queue
                const queueIndex = this.maskQueue.indexOf(id);
                if (queueIndex !== -1) {
                    this.maskQueue.splice(queueIndex, 1);
                }

                // Update active mask if needed
                if (this.activeMaskId === id) {
                    this.activeMaskId = this.maskQueue.length > 0 ? this.maskQueue[0] : null;
                }
            }
        } catch (error) {
            console.error(`Failed to remove mask ${id}:`, error);
        }
    }

    setBindGroupLayout(layout) {
        if (!layout) throw new Error('Invalid bind group layout');

        try {
            this.bindGroupLayout = layout;
            // Recreate bind groups for existing masks
            for (const mask of this.masks.values()) {
                mask.bindGroup = this.createBindGroup(mask);
            }
        } catch (error) {
            console.error('Failed to update bind group layout:', error);
            throw error;
        }
    }

    // Resource cleanup
    destroy() {
        try {
            // Cleanup all masks
            for (const mask of this.masks.values()) {
                mask.texture.texture?.destroy?.();
            }
            this.masks.clear();
            this.maskQueue = [];
            this.activeMaskId = null;
            
            // Clear references
            this.bindGroupLayout = null;
        } catch (error) {
            console.error('Error during MaskManager cleanup:', error);
        }
    }
}