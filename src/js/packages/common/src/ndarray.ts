export enum ArrayDType {
    Float32 = "float32",
    Int32 = "int32",
    Int64 = "int64",
    String = "string",
    Bool = "bool"
}

export class NDArray {
    public shape: number[];
    private data: any[];
    public dtype: ArrayDType;

    constructor(shape: number[], data: any[], dtype: ArrayDType) {
        this.shape = shape;
        this.dtype = dtype;

        const size = shape.reduce((a, b) => a * b, 1);

        if (!Object.values(ArrayDType).includes(dtype)) {
            throw new Error(`Unsupported dtype: ${dtype}`);
        }

        if (data.length !== size) {
            throw new Error(
                `Data length (${data.length}) does not match the size of the shape (${size})`
            );
        }
        this.data = data.map(item => this.castToDType(item));
    }

    private castToDType(value: any): any {
        switch (this.dtype) {
            case ArrayDType.Float32:
                return parseFloat(value);
            case ArrayDType.Int32:
                return parseInt(value, 10);
            case ArrayDType.Int64:
                return BigInt(value);
            case ArrayDType.String:
                return String(value);
            case ArrayDType.Bool:
                return Boolean(value);
            default:
                throw new Error(`Unsupported dtype: ${this.dtype}`);
        }
    }

    private computeIndex(indices: number[]): number {
        if (indices.length !== this.shape.length) {
            throw new Error(`Expected ${this.shape.length} indices, got ${indices.length}`);
        }
        let index = 0;
        let stride = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= this.shape[i]) {
                throw new Error(`Index ${indices[i]} out of bounds for dimension ${i}`);
            }
            index += indices[i] * stride;
            stride *= this.shape[i];
        }
        return index;
    }

    get(indices: number[]): any {
        const flatIndex = this.computeIndex(indices);
        return this.data[flatIndex];
    }

    set(indices: number[], value: any): void {
        const flatIndex = this.computeIndex(indices);
        this.data[flatIndex] = this.castToDType(value);
    }

    getShape(): number[] {
        return this.shape;
    }

    getDType(): ArrayDType {
        return this.dtype;
    }

    toArray(): any[] {
        return this.data.slice();
    }

    private formatForLogging(shape: number[], flatData: any[], offset: number = 0): string {
        if (shape.length === 1) {
            const start = offset;
            const end = start + shape[0];
            return `[${flatData.slice(start, end).join(", ")}]`;
        }

        const stride = shape.slice(1).reduce((a, b) => a * b, 1);
        const result: string[] = [];
        for (let i = 0; i < shape[0]; i++) {
            const subArray = this.formatForLogging(shape.slice(1), flatData, offset + i * stride);
            result.push(subArray);
        }
        return `[\n${result.map(r => "  " + r).join(",\n")}\n]`;
    }

    log(): void {
        console.log(this.formatForLogging(this.shape, this.data));
    }

    astype(newDType: ArrayDType): NDArray {
        if (!Object.values(ArrayDType).includes(newDType)) {
            throw new Error(`Unsupported dtype for casting: ${newDType}`);
        }

        const newData = this.data.map(value => {
            switch (newDType) {
                case ArrayDType.Float32:
                    return parseFloat(value);
                case ArrayDType.Int32:
                    return parseInt(value, 10);
                case ArrayDType.Int64:
                    return BigInt(value);
                case ArrayDType.String:
                    return String(value);
                case ArrayDType.Bool:
                    return Boolean(value);
                default:
                    throw new Error(`Unsupported dtype for casting: ${newDType}`);
            }
        });

        return new NDArray(this.shape, newData, newDType);
    }
}


const RESERVED_TYPES = ["string", "integer", "float", "Array", "NDContainer"];

export class DtypesManager {
    private dtypes: Record<string, object>;

    constructor(externalDtypes: Record<string, object> = {}) {
        this.dtypes = { ...externalDtypes };

        for (const dtype of Object.keys(this.dtypes)) {
            if (dtype.includes("[")) {
                throw new Error(`Invalid dtype name: ${dtype}`);
            }
        }
        for (const reserved of RESERVED_TYPES) {
            if (reserved in this.dtypes) {
                throw new Error(`Invalid dtype name: ${reserved}`);
            }
        }
    }

    getDtype(name: string): object {
        if (!(name in this.dtypes)) {
            throw new Error(`Unknown dtype: ${name}`);
        }
        return this.dtypes[name];
    }

    validateDtype(name: string, data: unknown): void {
        if (Array.isArray(data)) {
            for (const d of data) {
                this.validateDtype(name, d);
            }
        } else if (typeof data === "object" && data !== null) {
            this.validateJsonSchema(name, data as Record<string, unknown>);
        } else if (typeof data === "string") {
            if (name !== "string") {
                throw new TypeError(`Invalid data type, expected \`string\`, got \`${name}\``);
            }
        } else if (typeof data === "number" && Number.isInteger(data)) {
            if (name !== "integer") {
                throw new TypeError(`Invalid data type, expected \`integer\`, got \`${name}\``);
            }
        } else if (typeof data === "number") {
            if (name !== "float") {
                throw new TypeError(`Invalid data type, expected \`float\`, got \`${name}\``);
            }
        } else {
            throw new TypeError(`Invalid data type: ${typeof data}`);
        }
    }

    validateJsonSchema(name: string, data: Record<string, unknown>): void {
        const schema = this.getDtype(name);
        if (typeof schema !== "object" || schema === null) {
            throw new Error(`Invalid schema for dtype: ${name}`);
        }
        // Basic JSON schema validation: check required properties and types
        const s = schema as Record<string, any>;
        if (s.type === "object" && s.properties) {
            if (s.required) {
                for (const req of s.required as string[]) {
                    if (!(req in data)) {
                        throw new Error(`Missing required property: ${req}`);
                    }
                }
            }
        }
    }
}

export class NDContainer {
    public readonly shape: number[];
    public readonly data: unknown[];
    private readonly _dtype: string;
    private readonly dtypesManager: DtypesManager | null;

    constructor(data: unknown, dtype: string, dtypesManager: DtypesManager | null = null) {
        if (dtype.startsWith("Array[")) {
            throw new Error("NDContainer does not support Array dtype");
        }
        if (dtype.startsWith("NDContainer[")) {
            dtype = dtype.slice(12, -1);
        }

        this.data = Array.isArray(data) ? structuredClone(data) : [structuredClone(data)];

        if (dtypesManager) {
            dtypesManager.validateDtype(dtype, this.data);
        }
        this.dtypesManager = dtypesManager;
        this._dtype = dtype;
        this.shape = this.computeShape(this.data);
    }

    get dtype(): string {
        return this._dtype;
    }

    private computeShape(data: unknown): number[] {
        if (!Array.isArray(data) || data.length === 0) {
            return [];
        }
        return [data.length, ...this.computeShape(data[0])];
    }

    toArray(): unknown[] {
        return structuredClone(this.data);
    }
}