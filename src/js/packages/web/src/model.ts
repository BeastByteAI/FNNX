import { interfaces, LocalHandler, DtypesManager, Inputs, Outputs, DynamicAttributes } from "@fnnx/common";
import { TarExtractor } from "./tar.js";
import { ONNXOpV1 } from "./ops.js";

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

        this.manifest = retrieve_file_content('manifest.json', this.modelFiles) as interfaces.Manifest;
        this.ops = retrieve_file_content('ops.json', this.modelFiles) as interfaces.OpInstanceConfig[];
        this.variantConfig = retrieve_file_content('variant_config.json', this.modelFiles);

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
        const dtypesManager = new DtypesManager();
        const handlerConfig = { operators: op_implementations };
        const deviceMap: interfaces.DeviceMap = { accelerator: 'cpu', node_device_map: {}, variant_device_config: {} };
        this.handler = new LocalHandler(this.modelFiles, this.manifest, this.ops, this.variantConfig as interfaces.PipelineVariant, dtypesManager, deviceMap, handlerConfig);
        return await this.handler.warmup();
    }

    getManifest(): interfaces.Manifest {
        return JSON.parse(JSON.stringify(this.manifest));;
    }

    getMetadata(): Array<interfaces.MetaEntry> {
        return retrieve_file_content('meta.json', this.modelFiles) as Array<interfaces.MetaEntry>;
    }

    getDtypes(): Map<string, any> {
        return retrieve_file_content("dtypes.json", this.modelFiles) as Map<string, any>
    }

}

function extract(modelData: ArrayBuffer): interfaces.TarFileContent[] {
    const extractor = new TarExtractor(modelData);
    return extractor.extract();
}

const retrieve_file_content = (relpath: string, modelFiles: interfaces.TarFileContent[]): object => {
    const file = modelFiles.find(f => f.relpath === relpath);
    if (!file || !file.content) {
        throw new Error(`File ${relpath} not found in model`);
    }
    return JSON.parse(new TextDecoder().decode(file.content));
}
