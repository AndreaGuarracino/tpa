//! BPAF Visualizer - Minimal genome alignment dot plot viewer
//!
//! Usage: cargo run --example bpaf-viz -- <file.bpaf>

use eframe::egui;
use lib_bpaf::{AlignmentRecord, BpafReader};
use std::collections::HashMap;
use std::env;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone)]
struct Segment {
    abeg: u64,
    aend: u64,
    bbeg: u64,
    bend: u64,
    reverse: bool,
}

struct Plot {
    segments: Vec<Segment>,
    query_boundaries: Vec<(u64, String)>,
    target_boundaries: Vec<(u64, String)>,
    query_total_len: u64,
    target_total_len: u64,
}

struct ViewState {
    x: f64,
    y: f64,
    scale: f64,
}

impl ViewState {
    fn new(width: u64, height: u64) -> Self {
        let margin = 1.1;
        let scale = (width.max(height) as f64 * margin) / 800.0;
        Self {
            x: -(width as f64 * 0.05),
            y: -(height as f64 * 0.05),
            scale,
        }
    }

    fn genome_to_pixel(&self, gx: f64, gy: f64, canvas: egui::Rect) -> egui::Pos2 {
        let px = (gx - self.x) / self.scale;
        let py = (gy - self.y) / self.scale;
        egui::pos2(
            canvas.min.x + px as f32,
            canvas.max.y - py as f32,
        )
    }

    fn pixel_to_genome(&self, pos: egui::Pos2, canvas: egui::Rect) -> (f64, f64) {
        let px = (pos.x - canvas.min.x) as f64;
        let py = (canvas.max.y - pos.y) as f64;
        (
            self.x + px * self.scale,
            self.y + py * self.scale,
        )
    }

    fn zoom(&mut self, factor: f64, mouse: egui::Pos2, canvas: egui::Rect, plot: &Plot) {
        let (gx, gy) = self.pixel_to_genome(mouse, canvas);
        self.scale /= factor;
        let (new_gx, new_gy) = self.pixel_to_genome(mouse, canvas);
        self.x += gx - new_gx;
        self.y += gy - new_gy;
        self.clamp(plot);
    }

    fn pan(&mut self, delta: egui::Vec2, plot: &Plot) {
        self.x -= delta.x as f64 * self.scale;
        self.y += delta.y as f64 * self.scale;
        self.clamp(plot);
    }

    fn clamp(&mut self, plot: &Plot) {
        let max_x = plot.query_total_len as f64;
        let max_y = plot.target_total_len as f64;
        self.x = self.x.max(0.0).min(max_x);
        self.y = self.y.max(0.0).min(max_y);
    }
}

// ============================================================================
// Plot Construction
// ============================================================================

impl Plot {
    fn from_bpaf(path: &str) -> std::io::Result<Self> {
        println!("Loading BPAF file: {}", path);
        let mut reader = BpafReader::open(path)?;

        // Build sequence tables
        let mut query_seqs: HashMap<u64, (String, u64)> = HashMap::new();
        let mut target_seqs: HashMap<u64, (String, u64)> = HashMap::new();

        // Collect all records first
        let segments: Vec<AlignmentRecord> = reader.iter_records()
            .collect::<Result<Vec<_>, _>>()?;

        // Now access string table
        let string_table = reader.string_table()?;

        // Build sequence maps
        for record in &segments {
            let query_name = string_table.get(record.query_name_id)
                .unwrap_or("unknown").to_string();
            let query_len = string_table.get_length(record.query_name_id)
                .unwrap_or(record.query_end);

            let target_name = string_table.get(record.target_name_id)
                .unwrap_or("unknown").to_string();
            let target_len = string_table.get_length(record.target_name_id)
                .unwrap_or(record.target_end);

            query_seqs.entry(record.query_name_id)
                .or_insert((query_name, query_len));
            target_seqs.entry(record.target_name_id)
                .or_insert((target_name, target_len));
        }

        // Build cumulative boundaries
        let mut query_list: Vec<_> = query_seqs.into_iter().collect();
        query_list.sort_by_key(|(id, _)| *id);

        let mut target_list: Vec<_> = target_seqs.into_iter().collect();
        target_list.sort_by_key(|(id, _)| *id);

        let mut query_offsets = HashMap::new();
        let mut query_boundaries = Vec::new();
        let mut cumulative = 0u64;

        for (id, (name, len)) in &query_list {
            query_offsets.insert(*id, cumulative);
            query_boundaries.push((cumulative, name.clone()));
            cumulative += len;
        }
        let query_total_len = cumulative;
        query_boundaries.push((query_total_len, String::new()));

        let mut target_offsets = HashMap::new();
        let mut target_boundaries = Vec::new();
        cumulative = 0;

        for (id, (name, len)) in &target_list {
            target_offsets.insert(*id, cumulative);
            target_boundaries.push((cumulative, name.clone()));
            cumulative += len;
        }
        let target_total_len = cumulative;
        target_boundaries.push((target_total_len, String::new()));

        // Convert to genome-wide coordinates
        let segments: Vec<Segment> = segments.into_iter().map(|rec| {
            let q_offset = query_offsets[&rec.query_name_id];
            let t_offset = target_offsets[&rec.target_name_id];

            let reverse = rec.strand == '-';
            let (bbeg, bend) = if reverse {
                (t_offset + rec.target_end, t_offset + rec.target_start)
            } else {
                (t_offset + rec.target_start, t_offset + rec.target_end)
            };

            Segment {
                abeg: q_offset + rec.query_start,
                aend: q_offset + rec.query_end,
                bbeg,
                bend,
                reverse,
            }
        }).collect();

        println!("Loaded {} segments", segments.len());
        println!("Query: {} sequences, {} bp", query_boundaries.len() - 1, query_total_len);
        println!("Target: {} sequences, {} bp", target_boundaries.len() - 1, target_total_len);

        Ok(Plot {
            segments,
            query_boundaries,
            target_boundaries,
            query_total_len,
            target_total_len,
        })
    }

    fn query_visible(&self, view: &ViewState, canvas_width: f32, canvas_height: f32) -> Vec<&Segment> {
        let x_min = view.x;
        let x_max = view.x + canvas_width as f64 * view.scale;
        let y_min = view.y;
        let y_max = view.y + canvas_height as f64 * view.scale;

        self.segments.iter().filter(|seg| {
            let seg_x_min = seg.abeg.min(seg.aend) as f64;
            let seg_x_max = seg.abeg.max(seg.aend) as f64;
            let seg_y_min = seg.bbeg.min(seg.bend) as f64;
            let seg_y_max = seg.bbeg.max(seg.bend) as f64;

            seg_x_max >= x_min && seg_x_min <= x_max &&
            seg_y_max >= y_min && seg_y_min <= y_max
        }).collect()
    }
}

// ============================================================================
// GUI Application
// ============================================================================

struct BpafViz {
    plot: Plot,
    view: ViewState,
    drag_start: Option<egui::Pos2>,
    cursor_info: String,
    forward_color: egui::Color32,
    reverse_color: egui::Color32,
}

impl BpafViz {
    fn new(plot: Plot) -> Self {
        let view = ViewState::new(plot.query_total_len, plot.target_total_len);
        Self {
            plot,
            view,
            drag_start: None,
            cursor_info: String::new(),
            forward_color: egui::Color32::from_rgb(0, 200, 0),
            reverse_color: egui::Color32::from_rgb(200, 0, 0),
        }
    }

    fn find_sequence(&self, pos: u64, boundaries: &[(u64, String)]) -> (String, u64) {
        for i in 0..boundaries.len() - 1 {
            if pos >= boundaries[i].0 && pos < boundaries[i + 1].0 {
                let local_pos = pos - boundaries[i].0;
                return (boundaries[i].1.clone(), local_pos);
            }
        }
        (String::from("?"), 0)
    }

    fn render_canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(
            ui.available_size(),
            egui::Sense::click_and_drag(),
        );

        let canvas = response.rect;

        // Background
        painter.rect_filled(canvas, 0.0, egui::Color32::BLACK);

        // Draw boundaries (gray grid)
        let boundary_color = egui::Color32::from_gray(40);
        for (pos, _) in &self.plot.query_boundaries {
            if *pos > 0 {
                let p = self.view.genome_to_pixel(*pos as f64, self.view.y, canvas);
                if p.x >= canvas.min.x && p.x <= canvas.max.x {
                    painter.vline(p.x, canvas.y_range(), (1.0, boundary_color));
                }
            }
        }
        for (pos, _) in &self.plot.target_boundaries {
            if *pos > 0 {
                let p = self.view.genome_to_pixel(self.view.x, *pos as f64, canvas);
                if p.y >= canvas.min.y && p.y <= canvas.max.y {
                    painter.hline(canvas.x_range(), p.y, (1.0, boundary_color));
                }
            }
        }

        // Draw alignments
        let visible = self.plot.query_visible(&self.view, canvas.width(), canvas.height());
        for seg in visible {
            let p1 = self.view.genome_to_pixel(seg.abeg as f64, seg.bbeg as f64, canvas);
            let p2 = self.view.genome_to_pixel(seg.aend as f64, seg.bend as f64, canvas);
            let color = if seg.reverse { self.reverse_color } else { self.forward_color };
            painter.line_segment([p1, p2], (1.0, color));
        }

        // Handle interactions
        if let Some(pos) = response.hover_pos() {
            let (gx, gy) = self.view.pixel_to_genome(pos, canvas);
            let (q_name, q_pos) = self.find_sequence(gx as u64, &self.plot.query_boundaries);
            let (t_name, t_pos) = self.find_sequence(gy as u64, &self.plot.target_boundaries);
            self.cursor_info = format!(
                "Query: {} @ {} | Target: {} @ {} | Scale: {:.1} bp/px",
                q_name, q_pos, t_name, t_pos, self.view.scale
            );
        }

        // Pan
        if response.dragged() {
            if let Some(start) = self.drag_start {
                let delta = response.interact_pointer_pos().unwrap() - start;
                self.view.pan(delta, &self.plot);
                self.drag_start = response.interact_pointer_pos();
            } else {
                self.drag_start = response.interact_pointer_pos();
            }
        } else {
            self.drag_start = None;
        }

        // Zoom
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let factor = if scroll > 0.0 { 1.2 } else { 1.0 / 1.2 };
                let mouse = response.hover_pos().unwrap_or(canvas.center());
                self.view.zoom(factor, mouse, canvas, &self.plot);
            }
        }
    }
}

impl eframe::App for BpafViz {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Controls:");
                ui.label("Drag = Pan");
                ui.label("Scroll = Zoom");
                ui.separator();
                ui.colored_label(self.forward_color, "Forward");
                ui.colored_label(self.reverse_color, "Reverse");
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.label(&self.cursor_info);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_canvas(ui);
        });

        ctx.request_repaint();
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file.bpaf>", args[0]);
        std::process::exit(1);
    }

    let plot = Plot::from_bpaf(&args[1])?;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 900.0])
            .with_title("BPAF Visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "BPAF Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(BpafViz::new(plot)))),
    )?;

    Ok(())
}
