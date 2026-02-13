import { interfaces, LocalHandler, DtypesManager, Inputs, Outputs, DynamicAttributes, applyPatches } from "@fnnx/common";
import { TarExtractor } from "./tar.js";
import { ONNXOpV1 } from "./ops.js";

const MANIFEST_PATCH_PATTERN = /^manifest-[^/]+\.patch\.json$/;
const META_PATTERN = /^meta(-[^/]+)?\.json$/;

const op_implementations = {
    "ONNX_v1": ONNXOpV1
}

export class Model {

    private modelFiles: interfaces.TarFileContent[];
    private manifest: interfaces.Manifest;
    private handler: LocalHandler | null = null;
    private ops: interfaces.OpInstanceConfig[];
    private variantConfig: object;

    private constructor(modelFiles: interfaces.TarFileContent[]) {
        this.modelFiles = modelFiles;

        this.manifest = this.loadManifest();
        this.ops = retrieveFileContent('ops.json', this.modelFiles) as interfaces.OpInstanceConfig[];
        this.variantConfig = retrieveFileContent('variant_config.json', this.modelFiles);
    }

    static async fromPath(modelPath: string): Promise<Model> {
        const response = await fetch(modelPath);
        const arrayBuffer = await response.arrayBuffer();
        const modelFiles = extract(arrayBuffer);
        return new Model(modelFiles);
    }

    static async fromBuffer(modelData: ArrayBuffer): Promise<Model> {
        const modelFiles = extract(modelData);
        return new Model(modelFiles);
    }

    async compute(inputs: Inputs, dynamicAttributes: DynamicAttributes): Promise<Outputs> {
        if (this.handler === null) {
            throw new Error('Model handler is not initialized. Please call warmup() before compute().');
        }
        return await this.handler.compute(inputs, dynamicAttributes);
    }

    async warmup() {
        let externalDtypes: Record<string, object> = {};
        try {
            externalDtypes = retrieveFileContent('dtypes.json', this.modelFiles) as Record<string, object>;
        } catch {
            // dtypes.json may not exist
        }
        const dtypesManager = new DtypesManager(externalDtypes);
        const handlerConfig = { operators: op_implementations };
        const deviceMap: interfaces.DeviceMap = { accelerator: 'cpu', node_device_map: {}, variant_device_config: {} };
        this.handler = new LocalHandler(this.modelFiles, this.manifest, this.ops, this.variantConfig as interfaces.PipelineVariant, dtypesManager, deviceMap, handlerConfig);
        return await this.handler.warmup();
    }

    getManifest(): interfaces.Manifest {
        return JSON.parse(JSON.stringify(this.manifest));
    }

    getMetadata(): Array<interfaces.MetaEntry> {
        const entries: interfaces.MetaEntry[] = [];
        for (const file of this.modelFiles) {
            if (file.type === "file" && META_PATTERN.test(file.relpath)) {
                const content = parseFileContent(file);
                if (Array.isArray(content)) {
                    entries.push(...(content as interfaces.MetaEntry[]));
                }
            }
        }
        return entries;
    }

    getDtypes(): Record<string, any> {
        try {
            return retrieveFileContent("dtypes.json", this.modelFiles) as Record<string, any>;
        } catch {
            return {};
        }
    }

    private loadManifest(): interfaces.Manifest {
        let manifestData = retrieveFileContent('manifest.json', this.modelFiles) as Record<string, any>;

        const patchFiles = this.modelFiles
            .filter((f) => f.type === "file" && MANIFEST_PATCH_PATTERN.test(f.relpath))
            .sort((a, b) => a.relpath.localeCompare(b.relpath));

        if (patchFiles.length > 0) {
            const patches = patchFiles.map(
                (pf) => parseFileContent(pf) as Array<{ op: string; path: string; value?: any }>
            );
            manifestData = applyPatches(manifestData, patches);
        }

        return manifestData as unknown as interfaces.Manifest;
    }
}

function extract(modelData: ArrayBuffer): interfaces.TarFileContent[] {
    const extractor = new TarExtractor(modelData);
    return extractor.extract();
}

function parseFileContent(file: interfaces.TarFileContent): object {
    if (!file.content) {
        throw new Error(`File ${file.relpath} has no content`);
    }
    return JSON.parse(new TextDecoder().decode(file.content));
}

const retrieveFileContent = (relpath: string, modelFiles: interfaces.TarFileContent[]): object => {
    const file = modelFiles.find(f => f.relpath === relpath);
    if (!file || !file.content) {
        throw new Error(`File ${relpath} not found in model`);
    }
    return JSON.parse(new TextDecoder().decode(file.content));
}
