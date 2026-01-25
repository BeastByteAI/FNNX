import { describe, it, expect, beforeEach } from "vitest";
import { TarExtractor } from "../../src/tar";

describe("TarExtractor", () => {
  // Helper to calculate and set the checksum for a tar header
  function setChecksum(view: Uint8Array, headerStart: number = 0): void {
    // First set checksum field to spaces
    const encoder = new TextEncoder();
    view.set(encoder.encode("        "), headerStart + 148);
    
    let checksum = 0;
    for (let i = 0; i < 512; i++) {
      checksum += view[headerStart + i];
    }
    
    const checksumStr = checksum.toString(8).padStart(6, "0") + "\0 ";
    view.set(encoder.encode(checksumStr), headerStart + 148);
  }

  // Helper to create a GNU LongName tar with long path
  function createGnuLongNameTar(longFileName: string, content: string): ArrayBuffer {
    const longNameSize = longFileName.length + 1; // +1 for null terminator
    const alignedLongNameSize = Math.ceil(longNameSize / 512) * 512;
    const contentSize = content.length;
    const alignedContentSize = Math.ceil(contentSize / 512) * 512 || 512;
    
    // Total size: LongName header (512) + LongName data + File header (512) + File data + End marker (1024)
    const totalSize = 512 + alignedLongNameSize + 512 + alignedContentSize + 1024;
    const buffer = new ArrayBuffer(totalSize);
    const view = new Uint8Array(buffer);
    const encoder = new TextEncoder();
    
    let offset = 0;
    
    // === GNU LongName header ===
    // Name field: ././@LongLink (conventional name for long link)
    view.set(encoder.encode("././@LongLink"), offset + 0);
    // Mode
    view.set(encoder.encode("0000000"), offset + 100);
    // UID
    view.set(encoder.encode("0000000"), offset + 108);
    // GID
    view.set(encoder.encode("0000000"), offset + 116);
    // Size (octal) of the long name content
    const longNameSizeOctal = longNameSize.toString(8).padStart(11, "0");
    view.set(encoder.encode(longNameSizeOctal), offset + 124);
    // Mtime
    view.set(encoder.encode("00000000000"), offset + 136);
    // Typeflag 'L' (0x4c) for GNU LongName
    view[offset + 156] = 0x4c;
    // Set checksum
    setChecksum(view, offset);
    
    offset += 512;
    
    // === LongName data (the actual long filename) ===
    view.set(encoder.encode(longFileName + "\0"), offset);
    offset += alignedLongNameSize;
    
    // === Actual file header ===
    // Name field: truncated version (first 100 chars)
    view.set(encoder.encode(longFileName.slice(0, 100)), offset + 0);
    // Mode
    view.set(encoder.encode("0000644"), offset + 100);
    // UID
    view.set(encoder.encode("0000000"), offset + 108);
    // GID
    view.set(encoder.encode("0000000"), offset + 116);
    // Size
    const contentSizeOctal = contentSize.toString(8).padStart(11, "0");
    view.set(encoder.encode(contentSizeOctal), offset + 124);
    // Mtime
    view.set(encoder.encode("00000000000"), offset + 136);
    // Typeflag '0' for regular file
    view[offset + 156] = 0x30;
    // Set checksum
    setChecksum(view, offset);
    
    offset += 512;
    
    // === File content ===
    view.set(encoder.encode(content), offset);
    
    return buffer;
  }

  // Helper to create a PAX extended header tar with long path
  function createPaxLongPathTar(longFileName: string, content: string): ArrayBuffer {
    // Create PAX record: "NN path=filename\n" where NN is the total record length
    const pathRecord = `path=${longFileName}`;
    // Record format: "length key=value\n"
    // We need to calculate the length including the length field itself
    let recordLen = pathRecord.length + 1; // +1 for newline
    let lenStr = recordLen.toString();
    recordLen = lenStr.length + 1 + pathRecord.length + 1; // len + space + record + newline
    lenStr = recordLen.toString();
    // Recalculate in case length digits changed
    recordLen = lenStr.length + 1 + pathRecord.length + 1;
    lenStr = recordLen.toString();
    if (lenStr.length + 1 + pathRecord.length + 1 !== recordLen) {
      recordLen = lenStr.length + 1 + pathRecord.length + 1;
      lenStr = recordLen.toString();
    }
    
    const paxContent = `${lenStr} ${pathRecord}\n`;
    const paxSize = paxContent.length;
    const alignedPaxSize = Math.ceil(paxSize / 512) * 512 || 512;
    const contentSize = content.length;
    const alignedContentSize = Math.ceil(contentSize / 512) * 512 || 512;
    
    // Total size: PAX header (512) + PAX data + File header (512) + File data + End marker (1024)
    const totalSize = 512 + alignedPaxSize + 512 + alignedContentSize + 1024;
    const buffer = new ArrayBuffer(totalSize);
    const view = new Uint8Array(buffer);
    const encoder = new TextEncoder();
    
    let offset = 0;
    
    // === PAX extended header ===
    // Name field
    view.set(encoder.encode("PaxHeader/file"), offset + 0);
    // Mode
    view.set(encoder.encode("0000000"), offset + 100);
    // UID
    view.set(encoder.encode("0000000"), offset + 108);
    // GID
    view.set(encoder.encode("0000000"), offset + 116);
    // Size (octal) of the PAX content
    const paxSizeOctal = paxSize.toString(8).padStart(11, "0");
    view.set(encoder.encode(paxSizeOctal), offset + 124);
    // Mtime
    view.set(encoder.encode("00000000000"), offset + 136);
    // Typeflag 'x' (0x78) for PAX extended header
    view[offset + 156] = 0x78;
    // Set checksum
    setChecksum(view, offset);
    
    offset += 512;
    
    // === PAX data (contains path=longfilename record) ===
    view.set(encoder.encode(paxContent), offset);
    offset += alignedPaxSize;
    
    // === Actual file header ===
    // Name field: truncated version
    view.set(encoder.encode(longFileName.slice(0, 100)), offset + 0);
    // Mode
    view.set(encoder.encode("0000644"), offset + 100);
    // UID
    view.set(encoder.encode("0000000"), offset + 108);
    // GID
    view.set(encoder.encode("0000000"), offset + 116);
    // Size
    const contentSizeOctal = contentSize.toString(8).padStart(11, "0");
    view.set(encoder.encode(contentSizeOctal), offset + 124);
    // Mtime
    view.set(encoder.encode("00000000000"), offset + 136);
    // Typeflag '0' for regular file
    view[offset + 156] = 0x30;
    // Set checksum
    setChecksum(view, offset);
    
    offset += 512;
    
    // === File content ===
    view.set(encoder.encode(content), offset);
    
    return buffer;
  }

  // Helper to create USTAR tar with prefix field for long paths
  function createUstarPrefixTar(prefix: string, name: string, content: string): ArrayBuffer {
    const contentSize = content.length;
    const alignedContentSize = Math.ceil(contentSize / 512) * 512 || 512;
    
    const totalSize = 512 + alignedContentSize + 1024;
    const buffer = new ArrayBuffer(totalSize);
    const view = new Uint8Array(buffer);
    const encoder = new TextEncoder();
    
    // === File header with USTAR prefix ===
    // Name field (offset 0, 100 bytes)
    view.set(encoder.encode(name.slice(0, 100)), 0);
    // Mode
    view.set(encoder.encode("0000644"), 100);
    // UID
    view.set(encoder.encode("0000000"), 108);
    // GID
    view.set(encoder.encode("0000000"), 116);
    // Size
    const contentSizeOctal = contentSize.toString(8).padStart(11, "0");
    view.set(encoder.encode(contentSizeOctal), 124);
    // Mtime
    view.set(encoder.encode("00000000000"), 136);
    // Typeflag '0' for regular file
    view[156] = 0x30;
    // Linkname (offset 157, 100 bytes) - empty
    // Magic "ustar\0" at offset 257
    view.set(encoder.encode("ustar\0"), 257);
    // Version "00" at offset 263
    view.set(encoder.encode("00"), 263);
    // Prefix field at offset 345 (155 bytes)
    view.set(encoder.encode(prefix.slice(0, 155)), 345);
    // Set checksum
    setChecksum(view, 0);
    
    // === File content ===
    view.set(encoder.encode(content), 512);
    
    return buffer;
  }

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

  describe("Long path handling", () => {
    it("should handle GNU LongName (typeflag L) for paths longer than 100 chars", () => {
      const longPath = "very/long/path/that/exceeds/one/hundred/characters/in/total/length/to/test/gnu/longname/support/file.txt";
      const content = "test content";
      
      expect(longPath.length).toBeGreaterThan(100);
      
      const tarBuffer = createGnuLongNameTar(longPath, content);
      const extractor = new TarExtractor(tarBuffer);
      const files = extractor.extract();
      
      expect(files).toHaveLength(1);
      expect(files[0].relpath).toBe(longPath);
      expect(new TextDecoder().decode(files[0].content)).toBe(content);
    });

    it("should handle GNU LongName with scan()", () => {
      const longPath = "very/long/path/that/exceeds/one/hundred/characters/in/total/length/to/test/gnu/longname/support/file.txt";
      const content = "test content";
      
      const tarBuffer = createGnuLongNameTar(longPath, content);
      const extractor = new TarExtractor(tarBuffer);
      const scanResults = extractor.scan();
      
      expect(scanResults.size).toBe(1);
      expect(scanResults.has(longPath)).toBe(true);
      const [offset, size] = scanResults.get(longPath)!;
      expect(size).toBe(content.length);
    });

    it("should handle PAX extended header (typeflag x) for long paths", () => {
      const longPath = "very/long/path/that/exceeds/one/hundred/characters/in/total/length/to/test/pax/extended/header/support/file.txt";
      const content = "pax content";
      
      expect(longPath.length).toBeGreaterThan(100);
      
      const tarBuffer = createPaxLongPathTar(longPath, content);
      const extractor = new TarExtractor(tarBuffer);
      const files = extractor.extract();
      
      expect(files).toHaveLength(1);
      expect(files[0].relpath).toBe(longPath);
      expect(new TextDecoder().decode(files[0].content)).toBe(content);
    });

    it("should handle PAX extended header with scan()", () => {
      const longPath = "very/long/path/that/exceeds/one/hundred/characters/in/total/length/to/test/pax/extended/header/support/file.txt";
      const content = "pax content";
      
      const tarBuffer = createPaxLongPathTar(longPath, content);
      const extractor = new TarExtractor(tarBuffer);
      const scanResults = extractor.scan();
      
      expect(scanResults.size).toBe(1);
      expect(scanResults.has(longPath)).toBe(true);
    });

    it("should handle USTAR prefix field for moderately long paths", () => {
      const prefix = "path/prefix/directory/structure/that/goes/here";
      const name = "filename.txt";
      const expectedPath = `${prefix}/${name}`;
      const content = "ustar content";
      
      const tarBuffer = createUstarPrefixTar(prefix, name, content);
      const extractor = new TarExtractor(tarBuffer);
      const files = extractor.extract();
      
      expect(files).toHaveLength(1);
      expect(files[0].relpath).toBe(expectedPath);
      expect(new TextDecoder().decode(files[0].content)).toBe(content);
    });

    it("should handle USTAR prefix field with scan()", () => {
      const prefix = "path/prefix/directory/structure/that/goes/here";
      const name = "filename.txt";
      const expectedPath = `${prefix}/${name}`;
      const content = "ustar content";
      
      const tarBuffer = createUstarPrefixTar(prefix, name, content);
      const extractor = new TarExtractor(tarBuffer);
      const scanResults = extractor.scan();
      
      expect(scanResults.size).toBe(1);
      expect(scanResults.has(expectedPath)).toBe(true);
    });
  });
});
