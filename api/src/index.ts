import { Hono } from 'hono'
import { serveStatic } from 'hono/bun'
import { cors } from 'hono/cors'
import { readdir, mkdir, unlink } from 'node:fs/promises'

const app = new Hono()

// Enable CORS
app.use('/*', cors())

// Serve static files from the images directory
app.use('/images/*', serveStatic({ root: './' }))

// Get the first image from the images folder
app.get('/api/image', async (c) => {
  try {
    const files = await readdir('./images')
    if (files.length === 0) {
      return c.json({ error: 'No images available' }, 404)
    }
    
    // Filter out any hidden files (like .DS_Store)
    const imageFiles = files.filter(f => !f.startsWith('.'))
    if (imageFiles.length === 0) {
      return c.json({ error: 'No images available' }, 404)
    }

    const firstImage = imageFiles[0]
    return c.json({ 
      filename: firstImage,
      url: `/images/${firstImage}`
    })
  } catch (error) {
    return c.json({ error: 'Error reading images' }, 500)
  }
})

// Move image to yes/no folder
app.post('/api/categorize', async (c) => {
  try {
    const { filename, category } = await c.req.json()
    if (!filename || !['yes', 'no'].includes(category)) {
      return c.json({ error: 'Invalid parameters' }, 400)
    }

    const sourcePath = `./images/${filename}`
    const targetPath = `./${category}/${filename}`

    // Create directory if it doesn't exist
    try {
      await mkdir(`./${category}`, { recursive: true })
    } catch (error) {
      // Ignore if directory already exists
    }

    // Read source file
    const sourceFile = Bun.file(sourcePath)
    const contents = await sourceFile.arrayBuffer()
    
    // Write to target file
    await Bun.write(targetPath, contents)
    
    // Delete source file
    await unlink(sourcePath)

    return c.json({ success: true })
  } catch (error) {
    console.error('Error moving file:', error)
    return c.json({ error: 'Error moving image' }, 500)
  }
})

export default {
  port: 3009,
  fetch: app.fetch
}