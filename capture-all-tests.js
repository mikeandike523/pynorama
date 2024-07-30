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

async function getCurrentPage(browser) {
  const targets = browser.targets();
  for (let i = 0, I = targets.length; i < I; ++i) {
    const target = targets[i];
    const page = await target.page();
    if (page) {
      return page;
    }
  }
  return null;
}

async function processDirectory(browser, dirName) {
  const inputPath = path.join(INPUT_DIR, dirName);
  const htmlPath = path.join(inputPath, "arrangement", "arrangement.html");

  const page = await getCurrentPage(browser);

  await page.goto(`file://${path.resolve(htmlPath)}`);

  // set page zoom to 0.25
  await page.evaluate(() => {
    document.body.style.zoom = "10%";
  });

  // delay 1000 millis

  await new Promise((resolve) => setTimeout(resolve, 1000));

  // screenshow element with id svg-container
  const svgContainer = await page.$("#svg-container");

  // Screenshot the SVG container
  await svgContainer.screenshot({
    path: path.join(OUTPUT_DIR, `${dirName}.png`),
  });
}

async function main() {
  await fs.ensureDir(OUTPUT_DIR);
  await fs.emptyDir(OUTPUT_DIR);

  const directories = await fs.readdir(INPUT_DIR);
  // Step 3 & 4: Open browser and navigate to SVG
  const browser = await puppeteer.launch({
    // headless: "new",
    headless: false,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--start-maximized"],
    defaultViewport: null,
  });

  for (const dir of directories) {
    if ((await fs.stat(path.join(INPUT_DIR, dir))).isDirectory()) {
      await processDirectory(browser, dir);
    }
  }

  await browser.close();
}

await main();
