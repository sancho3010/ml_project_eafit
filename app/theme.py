"""Paleta y estilos compartidos por toda la app."""

# Paleta accesible (colorblind-friendly), tonos azules/verdes/rojos sobrios.
COLOR_PRIMARY = "#1f4e79"
COLOR_SECONDARY = "#5b8db8"
COLOR_ACCENT = "#2e8b57"
COLOR_WARNING = "#c87f0a"
COLOR_DANGER = "#a13a3a"
COLOR_NEUTRAL = "#6b7280"
COLOR_BG_SOFT = "#f4f6f8"

CSS = """
<style>
:root {
    --color-primary: #1f4e79;
    --color-secondary: #5b8db8;
    --color-accent: #2e8b57;
    --color-bg-soft: #f4f6f8;
    --color-text: #1e2a3a;
    --color-muted: #6b7280;
}
section[data-testid="stSidebar"] {
    background-color: var(--color-bg-soft);
}
.kpi-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.kpi-card .label {
    font-size: 0.85rem;
    color: var(--color-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.25rem;
}
.kpi-card .value {
    font-size: 2rem;
    font-weight: 600;
    color: var(--color-primary);
    line-height: 1.2;
}
.kpi-card .delta {
    font-size: 0.85rem;
    color: var(--color-muted);
    margin-top: 0.25rem;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--color-primary);
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--color-primary);
}
.muted {
    color: var(--color-muted);
    font-size: 0.9rem;
}

/* Convierte los st.page_link en botones discretos */
a[data-testid="stPageLink-NavLink"] {
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    background-color: var(--color-primary) !important;
    color: white !important;
    padding: 0.5rem 1.25rem !important;
    border-radius: 6px !important;
    text-decoration: none !important;
    font-weight: 500;
    border: 1px solid var(--color-primary);
    transition: background-color 0.15s, border-color 0.15s;
    margin-top: 0.75rem;
}
a[data-testid="stPageLink-NavLink"]:hover {
    background-color: #163a5c !important;
    border-color: #163a5c;
}
a[data-testid="stPageLink-NavLink"] p {
    color: white !important;
    margin: 0 !important;
}
</style>
"""
