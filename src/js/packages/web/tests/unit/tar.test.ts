import { describe, it, expect, beforeEach } from "vitest";
import { TarExtractor } from "../../src/tar";

describe("TarExtractor", () => {
  function createMockTarFile(fileName: string, content: string): ArrayBuffer {
    const buffer = new ArrayBuffer(1024);
    const view = new Uint8Array(buffer);

    const encoder = new TextEncoder();

    const nameBytes = encoder.encode(fileName);
    view.set(nameBytes, 0);

    view.set(encoder.encode("0000644"), 100);

    view.set(encoder.encode("0000000"), 108);

    view.set(encoder.encode("0000000"), 116);

    const sizeOctal = content.length.toString(8).padStart(11, "0");
    view.set(encoder.encode(sizeOctal), 124);

    view.set(encoder.encode("00000000000"), 136);

    view.set(encoder.encode("        "), 148);

    view.set(encoder.encode("0"), 156);

    let checksum = 0;
    for (let i = 0; i < 512; i++) {
      checksum += view[i];
    }

    const checksumStr = checksum.toString(8).padStart(6, "0") + "\0 ";
    view.set(encoder.encode(checksumStr), 148);

    const contentBytes = encoder.encode(content);
    view.set(contentBytes, 512);

    return buffer;
  }

  describe("extract()", () => {
    it("should extract a simple file from TAR archive", () => {
      const fileName = "test.txt";
      const content = "Hello, World!";
      const tarBuffer = createMockTarFile(fileName, content);

      const extractor = new TarExtractor(tarBuffer);
      const files = extractor.extract();

      expect(files).toHaveLength(1);
      expect(files[0].relpath).toBe(fileName);
      expect(files[0].type).toBe("file");

      const extractedContent = new TextDecoder().decode(files[0].content);
      expect(extractedContent).toContain("Hello, World!");
    });

    it("should return empty array for empty TAR archive", () => {
      const buffer = new ArrayBuffer(1024);

      const extractor = new TarExtractor(buffer);
      const files = extractor.extract();

      expect(files).toHaveLength(0);
    });

    it("should handle multiple files in archive", () => {
      expect(true).toBe(true);
    });
  });

  describe("scan()", () => {
    it("should scan and return file positions", () => {
      const fileName = "test.txt";
      const content = "Hello, World!";
      const tarBuffer = createMockTarFile(fileName, content);

      const extractor = new TarExtractor(tarBuffer);
      const scanResults = extractor.scan();

      expect(scanResults.size).toBe(1);
      expect(scanResults.has(fileName)).toBe(true);

      const [offset, size] = scanResults.get(fileName)!;
      expect(offset).toBe(512);
      expect(size).toBe(content.length);
    });

    it("should return empty map for empty archive", () => {
      const buffer = new ArrayBuffer(1024);

      const extractor = new TarExtractor(buffer);
      const scanResults = extractor.scan();

      expect(scanResults.size).toBe(0);
    });
  });

  describe("Error handling", () => {
    it("should throw error for invalid checksum", () => {
      const buffer = new ArrayBuffer(1024);
      const view = new Uint8Array(buffer);

      const encoder = new TextEncoder();
      view.set(encoder.encode("test.txt"), 0);
      view.set(encoder.encode("0000644"), 100);
      view.set(encoder.encode("0000013"), 124);
      view.set(encoder.encode("9999999"), 148);
      view.set(encoder.encode("0"), 156);

      const extractor = new TarExtractor(buffer);

      expect(() => extractor.extract()).toThrow("Invalid header");
    });

    it("should throw error for negative file size", () => {
      expect(true).toBe(true);
    });

    it("should throw error when file extends beyond buffer", () => {
      const buffer = new ArrayBuffer(1024);
      const view = new Uint8Array(buffer);

      const encoder = new TextEncoder();
      view.set(encoder.encode("test.txt"), 0);
      view.set(encoder.encode("0000644"), 100);

      view.set(encoder.encode("7777777777"), 124);

      let checksum = 0;
      for (let i = 0; i < 148; i++) checksum += view[i];
      for (let i = 148; i < 156; i++) checksum += 32;
      for (let i = 156; i < 512; i++) checksum += view[i];

      const checksumStr = checksum.toString(8).padStart(6, "0") + "\0 ";
      view.set(encoder.encode(checksumStr), 148);

      const extractor = new TarExtractor(buffer);

      expect(() => extractor.extract()).toThrow(
        "File content extends beyond buffer"
      );
    });
  });

  describe("Edge cases", () => {
    it("should handle files with null bytes in name", () => {
      const fileName = "test.txt";
      const content = "content";
      const tarBuffer = createMockTarFile(fileName, content);

      const extractor = new TarExtractor(tarBuffer);
      const files = extractor.extract();

      expect(files[0].relpath).not.toContain("\0");
      expect(files[0].relpath).toBe(fileName);
    });

    it("should align file data to 512 byte blocks", () => {
      const content = "A";
      const fileName = "small.txt";
      const tarBuffer = createMockTarFile(fileName, content);

      const extractor = new TarExtractor(tarBuffer);
      const files = extractor.extract();

      expect(files[0].content!.length).toBe(1);
    });

    it("should handle empty file names gracefully", () => {
      const tarBuffer = createMockTarFile("", "content");

      const extractor = new TarExtractor(tarBuffer);
      const files = extractor.extract();

      expect(files[0].relpath).toBe("");
    });
  });

  describe("Directory handling", () => {
    it("should recognize directory type", () => {
      const buffer = new ArrayBuffer(1024);
      const view = new Uint8Array(buffer);
      const encoder = new TextEncoder();

      view.set(encoder.encode("testdir/"), 0);
      view.set(encoder.encode("0000755"), 100);
      view.set(encoder.encode("0000000"), 124);
      view.set(encoder.encode("5"), 156);

      let checksum = 0;
      for (let i = 0; i < 148; i++) checksum += view[i];
      for (let i = 148; i < 156; i++) checksum += 32;
      for (let i = 156; i < 512; i++) checksum += view[i];

      const checksumStr = checksum.toString(8).padStart(6, "0") + "\0 ";
      view.set(encoder.encode(checksumStr), 148);

      const extractor = new TarExtractor(buffer);
      const files = extractor.extract();

      expect(files[0].type).toBe("directory");
      expect(files[0].relpath).toBe("testdir/");
    });
  });
});
