import { Hono } from "hono";
import { serveStatic } from "hono/bun";
import { cors } from "hono/cors";
import { readdir, mkdir, unlink } from "node:fs/promises";

const app = new Hono();

// Enable CORS
app.use("/*", cors());

// Serve static files from the images directory
app.use("/images/*", serveStatic({ root: "./" }));

// Modified get image endpoint to include model prediction
app.get("/api/image", async (c) => {
  try {
    const files = await readdir("./images");
    if (files.length === 0) {
      return c.json({ error: "No images available" }, 404);
    }

    const imageFiles = files.filter((f) => !f.startsWith("."));
    if (imageFiles.length === 0) {
      return c.json({ error: "No images available" }, 404);
    }

    const firstImage = imageFiles[0];
    const imagePath = `./images/${firstImage}`;

    return c.json({
      filename: firstImage,
      url: `/images/${firstImage}`,
    });
  } catch (error) {
    return c.json({ error: "Error reading images" }, 500);
  }
});

// Move image to yes/no folder
app.post("/api/categorize", async (c) => {
  try {
    const { filename, category } = await c.req.json();
    if (!filename || !["yes", "no"].includes(category)) {
      return c.json({ error: "Invalid parameters" }, 400);
    }

    const sourcePath = `./images/${filename}`;
    const targetPath = `./${category}/${filename}`;

    // Create directory if it doesn't exist
    try {
      await mkdir(`./${category}`, { recursive: true });
    } catch (error) {
      // Ignore if directory already exists
    }

    // Read source file
    const sourceFile = Bun.file(sourcePath);
    const contents = await sourceFile.arrayBuffer();

    // Write to target file
    await Bun.write(targetPath, contents);

    // Delete source file
    await unlink(sourcePath);

    return c.json({ success: true });
  } catch (error) {
    console.error("Error moving file:", error);
    return c.json({ error: "Error moving image" }, 500);
  }
});

// Add this new endpoint after the existing endpoints
app.get("/api/image/count", async (c) => {
  try {
    // Count files in each directory
    const getDirectoryCount = async (dir: string) => {
      try {
        const files = await readdir(`./${dir}`);
        return files.filter((f) => !f.startsWith(".")).length;
      } catch {
        return 0;
      }
    };

    const remaining = await getDirectoryCount("images");
    const yes = await getDirectoryCount("yes");
    const no = await getDirectoryCount("no");

    return c.json({ remaining, yes, no });
  } catch (error) {
    return c.json({ error: "Error counting images" }, 500);
  }
});

export default {
  port: 3009,
  fetch: app.fetch,
};
