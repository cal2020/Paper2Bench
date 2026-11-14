// Compile classic-script demo source while preserving shared component globals.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
if (args.length !== 4 || args[0] !== "--source" || args[2] !== "--output") {
  throw new Error("Usage: build_demo.mjs --source SNAPSHOT --output DIST_STAGE");
}
const source = path.resolve(args[1]);
const output = path.resolve(args[3]);
const allowedRemoteScripts = new Set([
  "https://unpkg.com/react@18.3.1/umd/react.development.js",
  "https://unpkg.com/react-dom@18.3.1/umd/react-dom.development.js",
  "https://unpkg.com/@babel/standalone@7.29.0/babel.min.js",
]);
await mkdir(path.join(output, "assets"), { recursive: true });

for (const page of ["index.html", "example.html"]) {
  const html = await readFile(path.join(source, "docs", page), "utf8");
  const chunks = [
    'import React from "react";',
    'import { createRoot } from "react-dom/client";',
    "const ReactDOM = { createRoot };",
  ];
  const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)];
  if (!scripts.length) throw new Error(`Missing demo scripts: ${page}`);
  for (const [, attributes, content] of scripts) {
    const src = attributes.match(/\bsrc\s*=\s*["']([^"']+)["']/i)?.[1];
    if (src) {
      if (/^(?:https?:)?\/\//i.test(src)) {
        if (!allowedRemoteScripts.has(src)) throw new Error(`Unsupported external application script in ${page}`);
        continue;
      }
      if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*\.(?:js|jsx)$/.test(src)) {
        throw new Error(`Unsupported local demo script in ${page}`);
      }
      chunks.push(`\n// Source: docs/${src}\n` + await readFile(path.join(source, "docs", src), "utf8"));
    } else {
      chunks.push(content);
    }
  }
  const result = await build({
    stdin: { contents: chunks.join("\n"), loader: "jsx", resolveDir: root, sourcefile: page + ".jsx" },
    bundle: true, platform: "browser", format: "iife", target: ["es2020"],
    define: { "process.env.NODE_ENV": '"production"' },
    minify: true, legalComments: "inline", charset: "utf8", write: false,
  });
  if (result.outputFiles.length !== 1) throw new Error("Unexpected demo bundle output");
  const bytes = result.outputFiles[0].contents;
  const hash = createHash("sha256").update(bytes).digest("hex").slice(0, 16);
  const filename = `assets/${path.basename(page, ".html")}-${hash}.js`;
  await writeFile(path.join(output, filename), bytes);
  const clean = html.replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/<\/body\s*>/i, `<script defer src="${filename}"></script>\n</body>`);
  if (/<script\b[^>]*src\s*=\s*["'](?:https?:)?\/\//i.test(clean) || /text\/babel/.test(clean)) {
    throw new Error("Production demo retains external JavaScript or runtime JSX compilation");
  }
  await writeFile(path.join(output, page), clean);
}
