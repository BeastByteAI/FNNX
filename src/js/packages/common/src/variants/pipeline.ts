import { BaseVariant } from './base';
import { DagComponent, dagCompute } from './common/dag';
import { OpIO, TarFileContent, DeviceMap, OpInstanceConfig } from '../interfaces';
import Registry from '../registry';
import { DtypesManager } from '../ndarray';
import { BaseOp } from '../ops/base';

interface PipelineNode {
    op_instance_id: string;
    inputs: string[];
    outputs: string[];
    extra_dynattrs?: Record<string, string>;
}

interface PipelineConfig {
    nodes: PipelineNode[];
}

class PipelineNodeInstance implements DagComponent {
    operator: BaseOp;
    inputs: string[];
    outputs: string[];
    inputSpecs: OpIO[];
    outputSpecs: OpIO[];
    extra_dynattrs: Record<string, string>;

    constructor(config: {
        operator: BaseOp;
        inputs: string[];
        outputs: string[];
        inputSpecs: OpIO[];
        outputSpecs: OpIO[];
        extra_dynattrs?: Record<string, string>;
    }) {
        this.operator = config.operator;
        this.inputs = config.inputs;
        this.outputs = config.outputs;
        this.inputSpecs = config.inputSpecs;
        this.outputSpecs = config.outputSpecs;
        this.extra_dynattrs = config.extra_dynattrs || {};
    }
}

export class Pipeline extends BaseVariant {
    private pipelineNodeInstances: PipelineNodeInstance[] = [];

    constructor(
        modelContent: TarFileContent[],
        ops: OpInstanceConfig[],
        variantConfig: Record<string, any>,
        config: {
            registry: Registry;
            deviceMap: DeviceMap;
            dtypesManager: DtypesManager;
        }
    ) {
        super(modelContent, ops, variantConfig, config);
        this.postInit();
    }

    protected postInit(): void {
        const config = this.variantConfig as PipelineConfig;
        this.pipelineNodeInstances = config.nodes.map(node => {
            const opInstance = this.opInstances.get(node.op_instance_id);
            if (!opInstance) {
                throw new Error(`Operation instance ${node.op_instance_id} not found`);
            }

            return new PipelineNodeInstance({
                operator: opInstance.operator,
                inputs: node.inputs,
                outputs: node.outputs,
                inputSpecs: opInstance.inputSpecs,
                outputSpecs: opInstance.outputSpecs,
                extra_dynattrs: node.extra_dynattrs || {}
            });
        });
    }

    private async nodeCompute(
        nodeInstance: PipelineNodeInstance,
        nodeInputs: any[],
        passthrough: Record<string, any>
    ): Promise<any> {
        // validateInputs(nodeInputs, nodeInstance.inputSpecs); // TODO
        return await nodeInstance.operator.compute(nodeInputs, passthrough.dynamic_attributes);
    }

    async compute(
        inputs: Record<string, any>,
        dynamicAttributes: Record<string, any>
    ): Promise<Record<string, any>> {
        const passthrough = {
            dynamic_attributes: dynamicAttributes
        };
        return await dagCompute(
            inputs,
            this.pipelineNodeInstances,
            (component, inputs, pass) => this.nodeCompute(component as PipelineNodeInstance, inputs, pass),
            (result) => result.value,
            passthrough
        );
    }
}