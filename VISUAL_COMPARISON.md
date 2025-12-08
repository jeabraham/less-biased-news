# Visual Comparison: Before and After

This document shows the visual differences between the old and new HTML output.

## Article Display

### BEFORE
```html
<li>
  [female_leader (Jyoti Gondek)] <a href='...' target='_blank'>Mayor Jyoti Gondek Announces New Climate Initiative</a>
  <img src='gondek.jpg' alt='' style='max-width:400px;'>
  <p>Calgary's Mayor Jyoti Gondek unveiled a comprehensive climate action plan...</p>
  <p>The initiative includes investments in renewable energy...</p>
</li>
```

**Issues:**
- Title not bold - hard to see article boundaries
- Status shown before title - inconsistent hierarchy
- No source or date information visible
- Only one image shown even if multiple women present
- No spacing between articles
- Standard paragraph spacing

### AFTER
```html
<li style='margin-bottom: 18pt;'>
  <a href='...' target='_blank' style='font-weight: bold;'>Mayor Jyoti Gondek Announces New Climate Initiative</a>
  <div style='font-size: 0.85em; color: #666; margin-top: 4pt;'>
    [female_leader (Jyoti Gondek)] | Calgary Herald | Jan 15, 2024
  </div>
  <img src='gondek.jpg' alt='Article image' style='max-width:800px;'>
  <img src='cityhall.jpg' alt='Article image' style='max-width:800px;'>
  <p style='margin-bottom: 6pt;'>Calgary's Mayor Jyoti Gondek unveiled a comprehensive climate action plan...</p>
  <p style='margin-bottom: 6pt;'>The initiative includes investments in renewable energy...</p>
</li>
```

**Improvements:**
- ✅ **Bold title** - Better visual hierarchy
- ✅ **Rich metadata** - Status, leader, source, and date in smaller gray font
- ✅ **Multiple images** - Shows all images with women (gondek.jpg + cityhall.jpg)
- ✅ **Larger images** - 800px for female_leader articles (400 * 2 multiplier)
- ✅ **Better spacing** - 6pt between paragraphs, 18pt between articles
- ✅ **Accessible** - Descriptive alt text

## Image Sizing Logic

### BEFORE (Hardcoded)
```python
if image_status in ("female", "female_majority", "female_prominent"):
    base = 400
elif image_status == "no_face":
    base = 200
else:
    base = 150

if article.get("status") == "female_leader":
    final = base * 2
else:
    final = base
```

### AFTER (Configurable)
```python
# Get config defaults or use hardcoded defaults
if cfg and "image_sizes" in cfg:
    img_cfg = cfg["image_sizes"]
    female_base = img_cfg.get("female_base", 400)
    no_face_base = img_cfg.get("no_face_base", 200)
    other_base = img_cfg.get("other_base", 150)
    leader_multiplier = img_cfg.get("leader_multiplier", 2)
else:
    female_base = 400
    no_face_base = 200
    other_base = 150
    leader_multiplier = 2

if image_status in FEMALE_IMAGE_STATUSES:
    base = female_base
# ... rest of logic
```

**Improvements:**
- ✅ User-configurable via config.yaml
- ✅ Uses constant for maintainability
- ✅ Backwards compatible with defaults

## Image Classification

### BEFORE
```python
if art["status"] == "female_leader":
    process_article_images(art, image_list)  # All images
else:
    process_article_images(art, image_list[:1])  # Only first image
```

**Problem:** Non-leader articles couldn't display multiple images even if they contained women.

### AFTER
```python
# Process all images to classify them for HTML display
process_article_images(art, image_list)  # All images, always
```

**Improvement:**
- ✅ All articles get all images classified
- ✅ HTML can display multiple images with women for any article
- ✅ More images available for fallback_image_female logic

## Configuration File

### NEW in config.yaml.example
```yaml
# Image sizing configuration for HTML output
image_sizes:
  female_base: 400          # Base size for images with female faces
  no_face_base: 200         # Base size for images with no faces
  other_base: 150           # Base size for other images (male or low confidence)
  leader_multiplier: 2      # Multiply image size by this for female_leader articles
```

**Benefits:**
- Users can customize image sizes without code changes
- Larger screens can use bigger images
- Smaller screens can reduce sizes for faster loading
- Easy to experiment with different layouts

## CSS Changes

### BEFORE
```css
img { float: right; margin: 0 0 1em 1em; max-width: 300px; }
```

**Problem:** CSS max-width: 300px would override all inline styles, limiting images to 300px.

### AFTER
```css
img { float: right; margin: 0 0 1em 1em; }
```

**Fix:** Removed conflicting max-width rule. Now inline styles control image size properly.

## Real-World Example

For an article about a female CEO with 3 images (CEO photo, team photo, office photo):

**BEFORE:**
- Shows only CEO photo at 400px
- No indication of source or date
- Title not bold

**AFTER:**
- Shows CEO photo at 800px (female leader multiplier applied)
- Shows team photo at 800px (if it contains women)
- Shows office photo at 800px (if it contains women)
- Displays: "[female_leader (Sarah Chen)] | Tech News | Jan 15, 2024"
- Bold title for easy scanning
- 18pt space before next article

## User Experience Improvements

1. **Better Scanability** - Bold titles and consistent spacing make it easy to scan through articles
2. **More Context** - Source and date help users evaluate content freshness and credibility
3. **Richer Visual Experience** - Multiple images provide better context for stories
4. **Better for Female Leadership** - Articles about women leaders get larger, more prominent images
5. **Professional Look** - Consistent formatting and typography create a polished appearance

## Performance Considerations

**Image Classification:**
- Before: 1 image analyzed for most articles
- After: Up to 3 images (max_images) analyzed for all articles
- Impact: Slight increase in processing time, but enables better HTML output

**HTML Size:**
- Before: ~1 image per article
- After: Variable (1-3 images depending on classification)
- Impact: Slightly larger HTML files, but significantly better visual presentation

The performance trade-off is worthwhile for the improved user experience and better representation of articles featuring women.
