import { globby } from 'globby';
import { resolve } from 'path';
import { minify } from 'terser';
import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';
import { hmrPlugin, presets } from 'vite-plugin-web-components-hmr';

// Custom plugin to minify the ES build

const full_reload_plugin = {
  name: 'full-reload',
  handleHotUpdate({ server }) {
    server.ws.send({ type: 'full-reload' });
    return [];
  }
};

function minifyBundles() {
  return {
    name: 'minifyBundles',
    async generateBundle(options, bundle) {
      for (let key in bundle) {
        if (bundle[key].type == 'chunk' && key.endsWith('.js')) {
          const minifyCode = await minify(bundle[key].code, {
            sourceMap: false,
            format: {
              comments: false
            },
            compress: {
              passes: 2
            },
            mangle: {
              toplevel: true,
              module: true
            }
          });
          bundle[key].code = minifyCode.code;
        }
      }
      return bundle;
    }
  };
}

async function getLibraryEntryPoints() {
  const files = await globby(resolve(__dirname, 'src/components/**/*.ts'));
  files.sort();
  return files.filter(file => !file.includes('app'));
}

function getFrontendHTMLInputs() {
  return {
    main: resolve(__dirname, 'index.html'),
    sessions: resolve(__dirname, 'sessions.html')
  };
}

export default defineConfig(async ({ command, mode }) => {
  if (command === 'serve') {
    // Development
    return {
      plugins: [
        hmrPlugin({
          include: ['./src/**/*.ts'],
          presets: [presets.lit]
        })
        // full_reload_plugin
      ],
      server: {}
    };
  } else if (command === 'build') {
    switch (mode) {
      case 'production': {
        // Production: standard web page (default mode)
        return {
          build: {
            outDir: 'dist',
            rollupOptions: {
              input: getFrontendHTMLInputs()
            }
          }
        };
      }

      case 'github': {
        // Production: github page (default mode)
        return {
          base: '/euphony/',
          build: {
            outDir: 'dist',
            rollupOptions: {
              input: getFrontendHTMLInputs()
            }
          },
          plugins: []
        };
      }

      case 'library': {
        // Production: library that can be imported in other apps
        return {
          publicDir: false,
          build: {
            lib: {
              // Could also be a dictionary or array of multiple entry points
              entry: [
                ...(await getLibraryEntryPoints()),
                resolve(__dirname, 'src/euphony.ts')
              ],
              name: 'euphony',
              formats: ['es']
            },
            outDir: 'lib',
            rollupOptions: {
              external: [],
              output: {
                generatedCode: {
                  constBindings: true
                },
                globals: {},
                chunkFileNames: `chunks/[name].js`,
                entryFileNames(chunkInfo) {
                  if (chunkInfo.facadeModuleId.includes('src/components/')) {
                    // Extract the folder name after 'src/components/' and use it for the output name
                    const folderName = chunkInfo.facadeModuleId
                      .split('src/components/')[1]
                      .split('/')[0];
                    return `components/${folderName}/[name].js`;
                  }
                  return `[name].js`;
                },
                manualChunks(id) {
                  // Check if the file path includes "@shoelace-style" to group into a single Shoelace chunk
                  if (id.includes('@shoelace-style')) {
                    return 'shoelace';
                  }

                  // CSS inline
                  if (id.includes('.css?inline')) {
                    if (id.includes('src/components/')) {
                      // Extract the folder name after 'src/components/' and use it for the output name
                      const folderName = id
                        .split('src/components/')[1]
                        .split('/')[0];
                      return `css/${folderName}`;
                    }
                    return 'css-inline';
                  }

                  // Larger third party libraries
                  if (id.includes('prismjs')) {
                    return 'prismjs';
                  }

                  if (id.includes('prismjs')) {
                    return 'prismjs';
                  }

                  if (id.includes('dompurify')) {
                    return 'dompurify';
                  }

                  if (id.includes('marked')) {
                    return 'marked';
                  }

                  // All other third party libraries
                  if (id.includes('node_modules')) {
                    // return id
                    //   .toString()
                    //   .split('node_modules/')[1]
                    //   .split('/')[1]
                    //   .toString();
                    return 'third-party';
                  }
                }
              }
            }
          },
          plugins: [dts(), minifyBundles()]
        };
      }

      default: {
        console.error(`Error: unknown production mode ${mode}`);
        return null;
      }
    }
  }
});
