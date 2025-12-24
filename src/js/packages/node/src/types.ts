export interface TarEntry {
    name: string;
    data: Buffer;
}

export interface TarExtractor {
    extract(input: unknown): Promise<TarEntry[]>;
}