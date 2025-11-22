// Base interfaces
export interface ModelIO {
    name: string;
    content_type: string;
    dtype: string;
    tags?: string[];
}

export interface JSONI extends ModelIO {
    content_type: "JSON";
}

export interface NDJSON extends ModelIO {
    content_type: "NDJSON";
    dtype: string;
    shape: (string | number)[];
}

export interface Var {
    name: string;
    description: string;
    tags?: string[];
}

export interface Manifest {
    variant: string;
    name?: string;
    version?: string;
    description?: string;
    producer_name: string;
    producer_version: string;
    producer_tags: string[];
    inputs: (NDJSON | JSONI)[];
    outputs: (NDJSON | JSONI)[];
    dynamic_attributes: Var[];
    env_vars: Var[];
}

export interface PipelineNode {
    op_instance_id: string;
    inputs: string[];
    outputs: string[];
    extra_dynattrs: Record<string, string>;
}

export interface PipelineVariant {
    nodes: PipelineNode[];
}

export interface OpIO {
    dtype: string;
    shape: (number | string)[];
}

export interface OpDynamicAttribute {
    name: string;
    defaultValue: string;
}

export interface OpInstanceConfig {
    id: string;  // Pattern: ^[a-zA-Z0-9_]+$
    op: string;
    inputs: OpIO[];
    outputs: OpIO[];
    attributes: Record<string, any>;
    dynamicAttributes: Record<string, OpDynamicAttribute>;
}

export interface DeviceConfig {
    accelerator: string;
    device_config?: Record<string, any>;
}

export interface DeviceMap {
    accelerator: string;
    node_device_map: Record<string, Record<string, any>>;
    variant_device_config?: Record<string, any> | string;
}

export interface TarFileContent {
    relpath: string;
    type: "file" | "directory";
    content: Uint8Array | null;
    fsPath: string | null;
}

export interface MetaEntry {
    id: string;
    producer: string;
    producer_version: string;
    producer_tags: string[];
    payload: Record<string, any>;
}

