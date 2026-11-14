/**
 * File: optimize_images.mjs
 *
 * Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin
 *
 * Description: Optimizes TACME author images for modern web delivery.
 *
 * License: MIT License
 */

import { mkdir, readdir, rm, stat } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import sharp from "sharp";

const ROOT = process.cwd();

const SOURCE_DIRECTORY = path.join(ROOT, "static", "images", "authors", "source");

const OUTPUT_DIRECTORY = path.join(ROOT, "static", "images", "authors", "generated");

const IMAGE_SIZE = 192;

const SUPPORTED_SOURCE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp"]);

const OUTPUT_EXTENSIONS = [".avif", ".webp", ".jpg"];

const CHECK_MODE = process.argv.includes("--check");

async function getSourceImages() {
    const entries = await readdir(SOURCE_DIRECTORY, {
        withFileTypes: true,
    });

    const sources = entries
        .filter((entry) => entry.isFile())
        .filter((entry) => SUPPORTED_SOURCE_EXTENSIONS.has(path.extname(entry.name).toLowerCase()))
        .map((entry) => path.join(SOURCE_DIRECTORY, entry.name))
        .sort();

    const stems = new Set();

    for (const sourcePath of sources) {
        const stem = path.parse(sourcePath).name;

        if (stems.has(stem)) {
            throw new Error(`Duplicate author image stem: ${stem}`);
        }

        stems.add(stem);
    }

    return sources;
}

function getOutputPaths(sourcePath) {
    const stem = path.parse(sourcePath).name;

    return {
        avif: path.join(OUTPUT_DIRECTORY, `${stem}.avif`),
        webp: path.join(OUTPUT_DIRECTORY, `${stem}.webp`),
        jpeg: path.join(OUTPUT_DIRECTORY, `${stem}.jpg`),
    };
}

function createImagePipeline(sourcePath) {
    return sharp(sourcePath).rotate().resize({
        width: IMAGE_SIZE,
        height: IMAGE_SIZE,
        fit: sharp.fit.cover,
        position: sharp.strategy.attention,
        kernel: sharp.kernel.mks2021,
    });
}

async function optimizeImage(sourcePath) {
    const outputPaths = getOutputPaths(sourcePath);

    const pipeline = createImagePipeline(sourcePath);

    await Promise.all([
        pipeline
            .clone()
            .avif({
                quality: 55,
                effort: 6,
            })
            .toFile(outputPaths.avif),

        pipeline
            .clone()
            .webp({
                quality: 80,
                effort: 6,
                preset: "photo",
            })
            .toFile(outputPaths.webp),

        pipeline
            .clone()
            .jpeg({
                quality: 82,
                mozjpeg: true,
            })
            .toFile(outputPaths.jpeg),
    ]);

    console.log(`Optimized ${path.basename(sourcePath)}`);
}

async function checkImage(sourcePath) {
    const sourceInfo = await stat(sourcePath);
    const outputPaths = getOutputPaths(sourcePath);

    for (const outputPath of Object.values(outputPaths)) {
        let outputInfo;

        try {
            outputInfo = await stat(outputPath);
        } catch {
            throw new Error(`Missing optimized image: ${path.relative(ROOT, outputPath)}`);
        }

        if (outputInfo.size === 0) {
            throw new Error(`Optimized image is empty: ${path.relative(ROOT, outputPath)}`);
        }

        if (outputInfo.mtimeMs < sourceInfo.mtimeMs) {
            throw new Error(`Optimized image is out of date: ${path.relative(ROOT, outputPath)}`);
        }
    }
}

async function buildImages(sourceImages) {
    await rm(OUTPUT_DIRECTORY, {
        recursive: true,
        force: true,
    });

    await mkdir(OUTPUT_DIRECTORY, {
        recursive: true,
    });

    for (const sourcePath of sourceImages) {
        await optimizeImage(sourcePath);
    }
}

async function checkImages(sourceImages) {
    for (const sourcePath of sourceImages) {
        await checkImage(sourcePath);
    }

    console.log(`${sourceImages.length} author image(s) are up to date.`);
}

async function main() {
    const sourceImages = await getSourceImages();

    if (sourceImages.length === 0) {
        console.log("No author source images found.");
        return;
    }

    if (CHECK_MODE) {
        await checkImages(sourceImages);
        return;
    }

    await buildImages(sourceImages);

    console.log(`${sourceImages.length} author image(s) optimized.`);
}

await main();
