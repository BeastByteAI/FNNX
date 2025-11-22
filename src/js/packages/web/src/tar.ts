import { interfaces } from "@fnnx/common";

export class TarExtractor {
    private offset = 0;
    private view: DataView;

    constructor(private buffer: ArrayBuffer) {
        this.view = new DataView(buffer);
    }

    private readString(length: number): string {
        const bytes = new Uint8Array(this.buffer, this.offset, length);
        const nullIndex = bytes.indexOf(0);
        const strLength = nullIndex !== -1 ? nullIndex : length;
        const decoder = new TextDecoder('ascii');
        const str = decoder.decode(bytes.slice(0, strLength));
        this.offset += length;
        return str;
    }

    private readOctal(length: number): number {
        const str = this.readString(length).trim();
        return str ? parseInt(str, 8) : 0;
    }

    private align512(size: number): number {
        const remainder = size % 512;
        return remainder ? size + (512 - remainder) : size;
    }

    private calculateChecksum(headerStart: number): number {
        let sum = 0;
        // Sum all bytes in the header, treating checksum field as spaces
        for (let i = 0; i < 512; i++) {
            if (i >= 148 && i < 156) {
                // Checksum field is treated as spaces (ASCII 32)
                sum += 32;
            } else {
                sum += this.view.getUint8(headerStart + i);
            }
        }
        return sum;
    }

    private parseHeader(): interfaces.TarFileContent | null {
        if (this.offset >= this.buffer.byteLength) {
            return null;
        }

        // Check for end of archive (zero block)
        let isZeroBlock = true;
        for (let i = 0; i < 512; i++) {
            if (this.view.getUint8(this.offset + i) !== 0) {
                isZeroBlock = false;
                break;
            }
        }
        if (isZeroBlock) {
            return null;
        }

        const originalOffset = this.offset;
        const name = this.readString(100);
        const mode = this.readOctal(8);
        const uid = this.readOctal(8);
        const gid = this.readOctal(8);
        const size = this.readOctal(12);
        const mtime = this.readOctal(12);
        const checksum = this.readOctal(8);
        const type = this.readString(1);
        const linkname = this.readString(100);

        // Validate checksum
        const calculatedChecksum = this.calculateChecksum(originalOffset);
        if (checksum !== calculatedChecksum) {
            throw new Error(`Invalid header`);
        }

        // Validate size is not negative
        if (size < 0) {
            throw new Error('Invalid file size in tar header');
        }

        // Validate offset + size doesn't exceed buffer
        if (this.offset + size > this.buffer.byteLength) {
            throw new Error('File content extends beyond buffer');
        }

        // Reset offset to start of data block
        this.offset = originalOffset + 512;

        // Read file data
        const data = new Uint8Array(this.buffer, this.offset, size);

        // Move offset to next header, aligned to 512 bytes
        this.offset += this.align512(size);

        return {
            relpath: name.replace(/\0/g, ''),
            content: data,
            type: type === '5' ? 'directory' : 'file',
            fsPath: null
        };
    }

    extract(): interfaces.TarFileContent[] {
        const files: interfaces.TarFileContent[] = [];
        let file: interfaces.TarFileContent | null;
        while ((file = this.parseHeader()) !== null) {
            files.push(file);
        }
        return files;
    }

    scan(): Map<string, [number, number]> {
        const results = new Map<string, [number, number]>();
        let scanOffset = 0;

        while (scanOffset < this.buffer.byteLength) {
            let isZeroBlock = true;
            for (let i = 0; i < 512; i++) {
                if (this.view.getUint8(scanOffset + i) !== 0) {
                    isZeroBlock = false;
                    break;
                }
            }
            if (isZeroBlock) break;

            const nameBytes = new Uint8Array(this.buffer, scanOffset, 100);
            const nullIndex = nameBytes.indexOf(0);
            const strLength = nullIndex !== -1 ? nullIndex : 100;
            const name = new TextDecoder('ascii').decode(nameBytes.slice(0, strLength));

            const sizeBytes = new Uint8Array(this.buffer, scanOffset + 124, 12);
            const sizeStr = new TextDecoder('ascii').decode(sizeBytes).trim();
            const size = sizeStr ? parseInt(sizeStr, 8) : 0;

            results.set(name.replace(/\0/g, ''), [scanOffset + 512, size]);

            scanOffset += 512 + this.align512(size);
        }

        return results;
    }
}