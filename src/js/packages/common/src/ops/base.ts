import { DeviceConfig, OpDynamicAttribute, OpIO, TarFileContent } from '../interfaces';
import { DtypesManager } from '../ndarray';

export interface OpOutput {
    value: any[];
    metadata?: Record<string, any>;
}

export abstract class BaseOp {
    protected static supportedDynamicAttributes: string[] = [];
    protected static requiredDynamicAttributes: string[] = [];

    protected dynamicAttributeMap: Record<string, OpDynamicAttribute>;
    protected warmedUp: boolean = false;
    protected artifacts: TarFileContent[];
    protected deviceConfig: DeviceConfig;
    protected attributes: Record<string, any>;
    protected inputSpecs: OpIO[];
    protected outputSpecs: OpIO[];
    protected dtypesManager: DtypesManager;

    constructor(
        artifacts: TarFileContent[],
        config: {
            attributes: Record<string, any>;
            dynamicAttributeMap: Record<string, OpDynamicAttribute>;
            deviceConfig: DeviceConfig;
            inputSpecs: OpIO[];
            outputSpecs: OpIO[];
            dtypesManager: DtypesManager;
        }
    ) {
        this.dynamicAttributeMap = config.dynamicAttributeMap;
        this.artifacts = artifacts;
        this.deviceConfig = config.deviceConfig;
        this.attributes = config.attributes;
        this.inputSpecs = config.inputSpecs;
        this.outputSpecs = config.outputSpecs;
        this.dtypesManager = config.dtypesManager;
    }

    abstract warmup(...args: any[]): Promise<BaseOp>;

    abstract compute(
        inputs: any[],
        dynamicAttributes: Record<string, any>,
        ...args: any[]
    ): Promise<OpOutput>;

    protected extractDynamicAttribute(
        dynamicAttributes: Record<string, any>
    ): Record<string, any> {
        const extracted: Record<string, any> = {};

        for (const [key, value] of Object.entries(this.dynamicAttributeMap)) {
            const sourceName = value.name;
            const defaultValue = value.defaultValue;
            const sourceValue = sourceName ? dynamicAttributes[sourceName] : undefined;
            const targetValue = sourceValue ?? defaultValue;
            extracted[key] = targetValue;
        }

        return extracted;
    }

    protected verifyRequiredDynamicAttributes(
        dynamicAttributesMap: Record<string, any>
    ): void {
        for (const key of (this.constructor as typeof BaseOp).requiredDynamicAttributes) {
            if (!(key in dynamicAttributesMap)) {
                throw new Error(`Missing required dynamic attribute: ${key}`);
            }
        }
    }

    protected isWarmedUp(): boolean {
        return this.warmedUp;
    }

    protected setWarmedUp(value: boolean): void {
        this.warmedUp = value;
    }
}

export type ConcreteOp = new (artifacts: TarFileContent[],
    config: {
        attributes: Record<string, any>;
        dynamicAttributeMap: Record<string, OpDynamicAttribute>;
        deviceConfig: DeviceConfig;
        inputSpecs: OpIO[];
        outputSpecs: OpIO[];
        dtypesManager: DtypesManager;
    }) => BaseOp;