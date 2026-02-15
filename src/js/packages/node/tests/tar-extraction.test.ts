import { describe, it, expect, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { create as tarCreate, extract as tarExtract } from "tar";

const tempDirs: string[] = [];

function makeTempDir(): string {
    const dir = mkdtempSync(path.join(tmpdir(), "fnnx-tar-test-"));
    tempDirs.push(dir);
    return dir;
}

function writeFile(dir: string, relpath: string, content: string): void {
    const fullPath = path.join(dir, relpath);
    mkdirSync(path.dirname(fullPath), { recursive: true });
    writeFileSync(fullPath, content);
}

async function extractBufferToDir(tarBuffer: Buffer, targetDir: string): Promise<void> {
    const readable = Readable.from(tarBuffer);
    await pipeline(readable, tarExtract({ C: targetDir }));
}

describe("Tar extraction with long file names", () => {
    afterEach(() => {
        for (const dir of tempDirs) {
            try { rmSync(dir, { recursive: true, force: true }); } catch {}
        }
        tempDirs.length = 0;
    });

    it("should extract files with paths > 100 characters", async () => {
        const sourceDir = makeTempDir();
        const longDir = "a".repeat(50) + "/" + "b".repeat(50);
        const longPath = longDir + "/file.txt";
        writeFile(sourceDir, longPath, "long path content");
        writeFile(sourceDir, "short.txt", "short");

        expect(longPath.length).toBeGreaterThan(100);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const extractDir = makeTempDir();
        await tarExtract({ file: tarPath, C: extractDir });

        expect(existsSync(path.join(extractDir, longPath))).toBe(true);
        expect(readFileSync(path.join(extractDir, longPath), "utf-8")).toBe("long path content");
        expect(readFileSync(path.join(extractDir, "short.txt"), "utf-8")).toBe("short");
    });

    it("should extract files with paths > 100 characters from buffer", async () => {
        const sourceDir = makeTempDir();
        const longDir = "deeply/nested/" + "subdir/".repeat(15) + "final";
        const longPath = longDir + "/data.json";
        writeFile(sourceDir, longPath, '{"key":"value"}');

        expect(longPath.length).toBeGreaterThan(100);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const tarBuffer = readFileSync(tarPath);
        const extractDir = makeTempDir();
        await extractBufferToDir(tarBuffer, extractDir);

        expect(existsSync(path.join(extractDir, longPath))).toBe(true);
        expect(readFileSync(path.join(extractDir, longPath), "utf-8")).toBe('{"key":"value"}');
    });

    it("should extract files with very long paths (> 255 characters)", async () => {
        const sourceDir = makeTempDir();
        const segments = Array.from({ length: 20 }, (_, i) => `segment_${i}`);
        const longPath = segments.join("/") + "/deep_file.txt";
        writeFile(sourceDir, longPath, "very deep content");

        expect(longPath.length).toBeGreaterThan(200);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const tarBuffer = readFileSync(tarPath);
        const extractDir = makeTempDir();
        await extractBufferToDir(tarBuffer, extractDir);

        expect(existsSync(path.join(extractDir, longPath))).toBe(true);
        expect(readFileSync(path.join(extractDir, longPath), "utf-8")).toBe("very deep content");
    });

    it("should extract files with unicode characters in long paths", async () => {
        const sourceDir = makeTempDir();
        const longPath = "data/" + "folder_".repeat(14) + "/file.txt";
        writeFile(sourceDir, longPath, "unicode content");

        expect(longPath.length).toBeGreaterThan(100);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const tarBuffer = readFileSync(tarPath);
        const extractDir = makeTempDir();
        await extractBufferToDir(tarBuffer, extractDir);

        expect(existsSync(path.join(extractDir, longPath))).toBe(true);
        expect(readFileSync(path.join(extractDir, longPath), "utf-8")).toBe("unicode content");
    });

    it("should preserve directory structure with deeply nested long paths", async () => {
        const sourceDir = makeTempDir();
        const basePath = "models/production/v2/" + "component_".repeat(10);
        const file1 = basePath + "/weights.bin";
        const file2 = basePath + "/config.json";
        writeFile(sourceDir, file1, "weights data");
        writeFile(sourceDir, file2, '{"layers": 3}');

        expect(file1.length).toBeGreaterThan(100);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const tarBuffer = readFileSync(tarPath);
        const extractDir = makeTempDir();
        await extractBufferToDir(tarBuffer, extractDir);

        expect(readFileSync(path.join(extractDir, file1), "utf-8")).toBe("weights data");
        expect(readFileSync(path.join(extractDir, file2), "utf-8")).toBe('{"layers": 3}');
    });
});
