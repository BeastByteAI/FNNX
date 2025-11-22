import type { Model as FnnxModel } from '../web/src/index.ts';
import type { TarExtractor as FnnxTarExtractor } from '../web/src/tar.ts';

declare global {
  interface Window {
    Model: typeof FnnxModel;
    TarExtractor: typeof FnnxTarExtractor;
    testResults?: Record<string, any>;
  }
}

export {};
