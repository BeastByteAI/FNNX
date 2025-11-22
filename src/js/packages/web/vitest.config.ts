import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,

    environment: 'jsdom',
    
    include: ['tests/unit/**/*.test.ts', 'tests/unit/**/*.spec.ts'],
    
    exclude: ['node_modules', 'dist', 'tests/fixtures'],
    
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'tests/',
        'dist/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/index.ts', 
      ],
      thresholds: {
        lines: 70,
        functions: 70,
        branches: 70,
        statements: 70,
      },
    },

    testTimeout: 30000,
    hookTimeout: 30000,
    
    retry: 1,
    
    watch: false,
  },
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@tests': path.resolve(__dirname, './tests'),
    },
  },
});