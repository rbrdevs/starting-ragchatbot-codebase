# Frontend Changes: Theme Toggle Button

## Overview

Added a dark/light theme toggle button to the Course Materials Assistant UI. The toggle allows users to switch between dark mode (default) and light mode, with their preference persisted via localStorage.

## Files Modified

### 1. `frontend/index.html`

Added theme toggle button HTML inside the `<body>` tag, positioned before the main container:

```html
<button id="themeToggle" class="theme-toggle" aria-label="Toggle dark/light theme" title="Toggle theme">
    <svg class="sun-icon">...</svg>
    <svg class="moon-icon">...</svg>
</button>
```

**Features:**
- Sun icon (displayed in dark mode) - clicking switches to light mode
- Moon icon (displayed in light mode) - clicking switches to dark mode
- Accessible with `aria-label` and `title` attributes
- Keyboard navigable (focusable button element)

### 2. `frontend/style.css`

#### New CSS Variables Added

Extended the CSS variable system with additional properties for comprehensive theming:

| Variable | Purpose |
|----------|---------|
| `--primary-light` | Light tint of primary color for backgrounds |
| `--text-muted` | Muted text for less emphasis |
| `--border-subtle` | Subtle borders for code blocks |
| `--user-message-text` | Text color for user messages |
| `--assistant-message-text` | Text color for assistant messages |
| `--shadow-lg` | Larger shadow for elevated elements |
| `--code-text` | Text color for code blocks |
| `--scrollbar-track` | Scrollbar track background |
| `--scrollbar-thumb` | Scrollbar thumb color |
| `--scrollbar-thumb-hover` | Scrollbar thumb hover color |
| `--error-bg`, `--error-text`, `--error-border` | Error state colors |
| `--success-bg`, `--success-text`, `--success-border` | Success state colors |
| `--link-color`, `--link-hover` | Link colors |

#### Light Theme Color Palette (WCAG AA Compliant)

| Variable | Dark Value | Light Value | Notes |
|----------|------------|-------------|-------|
| `--background` | `#0f172a` | `#f8fafc` | Warm off-white |
| `--surface` | `#1e293b` | `#ffffff` | Pure white surfaces |
| `--surface-hover` | `#334155` | `#f1f5f9` | Subtle hover state |
| `--text-primary` | `#f1f5f9` | `#0f172a` | ~15:1 contrast ratio |
| `--text-secondary` | `#94a3b8` | `#475569` | ~7:1 contrast ratio |
| `--text-muted` | `#64748b` | `#64748b` | ~4.5:1 contrast ratio |
| `--border-color` | `#334155` | `#cbd5e1` | Visible borders |
| `--border-subtle` | `#1e293b` | `#e2e8f0` | Subtle borders |
| `--assistant-message` | `#374151` | `#f1f5f9` | Light gray bubbles |
| `--welcome-bg` | `#1e3a5f` | `#eff6ff` | Blue-tinted welcome |
| `--welcome-border` | `#3b82f6` | `#bfdbfe` | Softer blue border |
| `--code-bg` | `rgba(0,0,0,0.3)` | `#f1f5f9` | Light gray code blocks |
| `--code-text` | `#e2e8f0` | `#1e293b` | High contrast code |
| `--scrollbar-thumb` | `#475569` | `#cbd5e1` | Visible scrollbars |
| `--error-text` | `#f87171` | `#dc2626` | Darker red for light bg |
| `--success-text` | `#4ade80` | `#16a34a` | Darker green for light bg |
| `--link-color` | `#60a5fa` | `#2563eb` | Accessible link colors |

#### Accessibility Improvements

- **Text Contrast**: All text colors meet WCAG AA standards (minimum 4.5:1 for normal text)
- **Primary Text**: ~15:1 contrast ratio on white background
- **Secondary Text**: ~7:1 contrast ratio on white background
- **Muted Text**: ~4.5:1 contrast ratio (meets minimum)
- **Error/Success Colors**: Adjusted for visibility on light backgrounds

#### New Styles Added

- `.theme-toggle` - Fixed position button in top-right corner
- Icon visibility toggling based on current theme
- Hover and focus states with smooth transitions
- Responsive adjustments for mobile devices

**Button Styling:**
- Fixed position: top-right corner
- Circular shape (44px diameter, 40px on mobile)
- Smooth 0.3s transitions for all state changes
- Scale animation on hover/active
- Focus ring for accessibility
- Icon rotation on hover

### 3. `frontend/script.js`

**New Functions:**
- `initializeTheme()` - Reads saved theme from localStorage or system preference
- `setTheme(theme)` - Applies theme by setting/removing `data-theme` attribute
- `toggleTheme()` - Switches between light and dark themes

**Theme Logic:**
1. On page load (before DOM ready), `initializeTheme()` runs to prevent flash
2. Checks localStorage for saved preference
3. Falls back to system preference (`prefers-color-scheme`)
4. Defaults to dark theme if no preference found
5. Theme changes are immediately persisted to localStorage

**Event Listener:**
- Added click handler for `themeToggle` button in `setupEventListeners()`

## Usage

Users can:
1. Click the toggle button in the top-right corner
2. Use keyboard navigation (Tab to focus, Enter/Space to toggle)
3. Their preference is automatically saved and restored on page reload

## Accessibility Features

- Button has `aria-label="Toggle dark/light theme"` for screen readers
- Keyboard navigable with visible focus state
- `title` attribute provides tooltip on hover
- Icons change based on current theme to indicate action
- All text meets WCAG AA contrast requirements
- Focus rings clearly visible in both themes
- Error and success states have appropriate contrast

## Browser Support

- Uses CSS custom properties (CSS variables)
- Uses `localStorage` for persistence
- Uses `matchMedia` for system preference detection
- Supported in all modern browsers (Chrome, Firefox, Safari, Edge)

## Design Decisions

1. **Dark theme as default**: Matches the original design
2. **System preference fallback**: Respects user's OS-level preference
3. **Warm whites for light theme**: Reduces eye strain compared to pure white
4. **Softer shadows in light mode**: More natural appearance
5. **Visible scrollbars**: Ensures usability in both themes
6. **Consistent primary color**: Blue remains the accent in both themes
