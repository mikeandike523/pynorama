import puppeteer from "puppeteer";
import fs from "fs-extra";
import path from "path";
import { parseString } from "xml2js";
import { promisify } from "util";

const parseXml = promisify(parseString);

const INPUT_DIR = "./testing/output/test-sets";
const OUTPUT_DIR = "./testing/output/screenshots";

async function getSvgDimensions(filePath) {
  const svgContent = await fs.readFile(filePath, "utf-8");
  const result = await parseXml(svgContent);
  const { width, height } = result.svg.$;
  return { width: parseInt(width), height: parseInt(height) };
}

async function processDirectory(dirName) {
  const inputPath = path.join(INPUT_DIR, dirName);
  const outputPath = path.join(OUTPUT_DIR, dirName);
  const svgPath = path.join(inputPath, "output.svg");

  // Step 1: Create or clear output directory
  await fs.ensureDir(outputPath);
  await fs.emptyDir(outputPath);

  // Step 2: Get SVG dimensions
  const { width, height } = await getSvgDimensions(svgPath);

  // Step 3 & 4: Open browser and navigate to SVG
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width, height });
  await page.goto(`file://${path.resolve(svgPath)}`);

  // Step 5: Take screenshot
  await page.screenshot({
    path: path.join(outputPath, `${dirName}.png`),
    fullPage: true,
  });

  // Step 6: Close browser
  await browser.close();
}

async function main() {
  const directories = await fs.readdir(INPUT_DIR);

  for (const dir of directories) {
    if ((await fs.stat(path.join(INPUT_DIR, dir))).isDirectory()) {
      await processDirectory(dir);
    }
  }
}

await main()
