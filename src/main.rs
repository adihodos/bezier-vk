use std::{ffi::CStr, mem::size_of};

use ash::vk::{
    BlendFactor, BlendOp, BufferUsageFlags, ClearColorValue, ClearValue, ColorComponentFlags,
    CullModeFlags, DescriptorBufferInfo, DescriptorSet, DescriptorSetAllocateInfo,
    DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, DeviceSize,
    DynamicState, Extent2D, FrontFace, GraphicsPipelineCreateInfo, IndexType, MemoryPropertyFlags,
    Offset2D, PipelineBindPoint, PipelineColorBlendAttachmentState,
    PipelineColorBlendStateCreateInfo, PipelineDynamicStateCreateInfo,
    PipelineInputAssemblyStateCreateInfo, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
    PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
    PrimitiveTopology, Rect2D, RenderPassBeginInfo, SampleCountFlags, ShaderStageFlags,
    SubpassContents, VertexInputAttributeDescription, VertexInputBindingDescription,
    VertexInputRate, Viewport, WriteDescriptorSet,
};

use imgui::WindowFlags;
use ui::UiBackend;
use vulkan_renderer::{
    compile_shader_from_file, FrameRenderContext, UniqueBuffer, UniqueBufferMapping,
    UniquePipeline, VulkanState,
};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

mod ui;
mod vulkan_renderer;

use crate::vulkan_renderer::WindowSystemIntegration;

fn main() {
    let _logger = flexi_logger::Logger::with(
        flexi_logger::LogSpecification::builder()
            .default(flexi_logger::LevelFilter::Debug)
            .build(),
    )
    .adaptive_format_for_stderr(flexi_logger::AdaptiveFormat::Detailed)
    .start()
    .unwrap_or_else(|e| {
        panic!("Failed to start the logger {}", e);
    });

    let event_loop = EventLoop::new();
    let primary_monitor = event_loop
        .primary_monitor()
        .expect("Failed to obtain primary monitor");

    let monitor_size = primary_monitor.size();
    log::info!("{:?}", primary_monitor);

    let window = WindowBuilder::new()
        .with_title("Fractal Explorer (with Rust + Vulkan)")
        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(Some(
            primary_monitor,
        ))))
        .with_inner_size(monitor_size)
        .build(&event_loop)
        .unwrap();

    log::info!("Main window surface size {:?}", window.inner_size());

    window
        .set_cursor_position(PhysicalPosition::new(
            window.inner_size().width / 2,
            window.inner_size().height / 2,
        ))
        .expect("Failed to center cursor ...");

    let mut fractal_sim = FractalSimulation::new(&window);

    event_loop.run(move |event, _, control_flow| {
        fractal_sim.handle_event(&window, event, control_flow);
    });
}

pub struct InputState<'a> {
    pub window: &'a winit::window::Window,
    pub event: &'a winit::event::WindowEvent<'a>,
    pub control_down: bool,
    pub cursor_pos: (f32, f32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, enum_iterator::Sequence)]
enum FractalType {
    Mandelbrot,
    Julia,
}

struct FractalSimulation {
    canvas: Canvas,
    ui: UiBackend,
    control_down: bool,
    left_mouse_down: bool,
    cursor_pos: (f32, f32),
    vks: std::pin::Pin<Box<VulkanState>>,
    control_pts: Vec<Vec2F32>,
    grabbed: Option<usize>,
}

impl FractalSimulation {
    const CP_OUTER_RADIUS: f32 = 16f32;
    const CP_INNER_RADIUS: f32 = 12f32;
    const CP_COUNT: usize = 4;
    const BE_CURVE_WIDTH: f32 = 8f32;
    const CP_LINE_WIDTH: f32 = 4f32;
    const CP_COLOR: u32 = minmath::color_palette::material_design::GREY_900;
    const BE_NORMAL_LENGTH: f32 = 32f32;
    const BE_NORMAL_WIDTH: f32 = 2f32;
    const BE_EVALPT_RADIUS: f32 = 8f32;
    const BE_EVALPT_COLOR: u32 = minmath::color_palette::material_design::GREEN_A700;
    const BE_NORMAL_COLOR: u32 = minmath::color_palette::material_design::RED_A700;
    const BE_DIRECTION_COLOR: u32 = minmath::color_palette::material_design::INDIGO_A700;
    const BE_CURVE_COLOR: u32 = minmath::color_palette::material_design::DEEP_ORANGE_900;

    fn new(window: &winit::window::Window) -> FractalSimulation {
        let cursor_pos = (
            (window.inner_size().width / 2) as f32,
            (window.inner_size().height / 2) as f32,
        );

        use winit::platform::x11::WindowExtX11;
        let wsi = WindowSystemIntegration {
            native_disp: window.xlib_display().unwrap(),
            native_win: window.xlib_window().unwrap(),
        };

        log::info!("Cursor initial position {:?}", cursor_pos);

        let mut vks = Box::pin(VulkanState::new(wsi).expect("Failed to initialize vulkan ..."));
        vks.begin_resource_loading();

        let ui = UiBackend::new(window, &mut vks, ui::HiDpiMode::Default);
        let canvas = Canvas::new(&mut vks);

        vks.end_resource_loading();

        FractalSimulation {
            canvas,
            left_mouse_down: false,
            control_down: false,
            cursor_pos,
            vks,
            ui,
            control_pts: Vec::new(),
            grabbed: None,
        }
    }

    fn draw_bezier(&mut self) {
        use minmath::color_palette::material_design;

        self.control_pts.windows(2).for_each(|cps| {
            self.canvas.add_line(
                cps[0],
                cps[1],
                Self::CP_LINE_WIDTH,
                Self::CP_COLOR.into(),
                Self::CP_COLOR.into(),
            );
        });

        if self.control_pts.len() == Self::CP_COUNT {
            let b = CubicBezier {
                ctrl_points: [
                    self.control_pts[0],
                    self.control_pts[1],
                    self.control_pts[2],
                    self.control_pts[3],
                ],
            };

            let steps = 64u32;
            let t_step = 1f32 / steps as f32;

            let mut verts = Vec::new();
            b.subdivide(&mut verts);

            verts.windows(2).for_each(|pts| {
                self.canvas.add_line(
                    pts[0],
                    pts[1],
                    Self::BE_CURVE_WIDTH,
                    Self::BE_CURVE_COLOR.into(),
                    Self::BE_CURVE_COLOR.into(),
                );
            });

            (0..=steps / 4).for_each(|t| {
                let t = t as f32 * t_step * 4f32;
                let p = b.eval(t);

                let d = b.derivative(t);
                let n = b.normal(d);

                self.canvas.add_line(
                    p,
                    p + normalize(d) * Self::BE_NORMAL_LENGTH,
                    Self::BE_NORMAL_WIDTH,
                    Self::BE_DIRECTION_COLOR.into(),
                    Self::BE_DIRECTION_COLOR.into(),
                );

                self.canvas.add_line(
                    p,
                    p + n * Self::BE_NORMAL_LENGTH,
                    Self::BE_NORMAL_WIDTH,
                    Self::BE_NORMAL_COLOR.into(),
                    Self::BE_NORMAL_COLOR.into(),
                );

                self.canvas
                    .add_circle(Self::BE_EVALPT_RADIUS, p, Self::BE_EVALPT_COLOR.into());
            });
        }

        let colors: [RGBAColor; 4] = [
            material_design::RED_A700.into(),
            material_design::GREEN_A700.into(),
            material_design::BLUE_A700.into(),
            material_design::YELLOW_A700.into(),
        ];

        let outer_ring_color: RGBAColor = material_design::GREY_500.into();

        self.control_pts
            .iter()
            .zip(colors.iter())
            .for_each(|(&cp, &col)| {
                self.canvas.add_double_circle(
                    Self::CP_INNER_RADIUS,
                    Self::CP_OUTER_RADIUS,
                    cp,
                    col,
                    outer_ring_color,
                )
            });
    }

    fn begin_rendering(&mut self) -> FrameRenderContext {
        let img_size = self.vks.ds.surface.image_size;
        let frame_context = self.vks.begin_rendering(img_size);

        let render_area = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: frame_context.fb_size,
        };

        use minmath::color_palette;
        let c: minmath::colors::RGBAColorF32 = color_palette::basic::BLACK.into();

        unsafe {
            self.vks.ds.device.cmd_begin_render_pass(
                frame_context.cmd_buff,
                &RenderPassBeginInfo::builder()
                    .framebuffer(frame_context.framebuffer)
                    .render_area(render_area)
                    .render_pass(self.vks.renderpass)
                    .clear_values(&[ClearValue {
                        color: ClearColorValue {
                            float32: [c.r, c.g, c.b, c.a],
                        },
                    }]),
                SubpassContents::INLINE,
            );
        }

        frame_context
    }

    fn end_rendering(&mut self, frame_context: &FrameRenderContext) {
        unsafe {
            self.vks
                .ds
                .device
                .cmd_end_render_pass(frame_context.cmd_buff);
        }

        self.vks.end_rendering();
    }

    fn setup_ui(&mut self, window: &winit::window::Window) {
        self.draw_bezier();

        let ui = self.ui.new_frame(window);

        self.control_pts.iter().for_each(|&cp| {
            let pos = cp + Self::CP_OUTER_RADIUS * Vec2F32 { x: 1f32, y: 1f32 };
            ui.window(format!("({}, {})", pos.x, pos.y))
                .position([pos.x, pos.y], imgui::Condition::Always)
                .flags(
                    WindowFlags::NO_INPUTS
                        | WindowFlags::ALWAYS_AUTO_RESIZE
                        | WindowFlags::NO_DECORATION,
                )
                .build(|| {
                    ui.text(format!("({:3.3}, {:3.3})", cp.x, cp.y));
                });
        });
    }

    fn handle_window_event(
        &mut self,
        _window: &winit::window::Window,
        event: &winit::event::WindowEvent,
        control_flow: &mut winit::event_loop::ControlFlow,
    ) {
        match *event {
            WindowEvent::CloseRequested => {
                control_flow.set_exit();
            }

            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode:
                            vk @ Some(VirtualKeyCode::Up)
                            | vk @ Some(VirtualKeyCode::Escape)
                            | vk @ Some(VirtualKeyCode::Down),
                        ..
                    },
                ..
            } => match vk {
                Some(VirtualKeyCode::Escape) => control_flow.set_exit(),
                _ => {}
            },

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos.0 = position.x as f32;
                self.cursor_pos.1 = position.y as f32;

                self.grabbed.as_ref().map(|&i| {
                    self.control_pts[i].x = self.cursor_pos.0;
                    self.control_pts[i].y = self.cursor_pos.1;
                });
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.control_down = mods.ctrl();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                self.left_mouse_down =
                    state == ElementState::Pressed && button == MouseButton::Left;

                if self.left_mouse_down {
                    if self.control_down && self.control_pts.len() < Self::CP_COUNT {
                        self.control_pts.push(Vec2F32 {
                            x: self.cursor_pos.0,
                            y: self.cursor_pos.1,
                        });
                    }

                    self.grabbed = self.control_pts.iter().position(|cp| {
                        let v = Vec2F32 {
                            x: self.cursor_pos.0,
                            y: self.cursor_pos.1,
                        } - *cp;
                        use minmath::vec2::dot;
                        dot(v, v) <= 15f32 * 15f32
                    });
                } else {
                    self.grabbed = None;
                }
            }

            WindowEvent::Resized(_) => {}

            _ => {}
        }
    }

    fn handle_event(
        &mut self,
        window: &winit::window::Window,
        event: Event<()>,
        control_flow: &mut winit::event_loop::ControlFlow,
    ) {
        control_flow.set_poll();

        match event {
            Event::WindowEvent {
                event: ref win_event,
                ..
            } => {
                let wants_input = self.ui.handle_event(window, &event);
                if !wants_input {
                    self.handle_window_event(window, win_event, control_flow);
                }
            }

            Event::MainEventsCleared => {
                let frame_context = self.begin_rendering();

                self.setup_ui(window);
                self.canvas.draw(&self.vks, &frame_context);
                self.ui.draw_frame(&self.vks, &frame_context);

                self.end_rendering(&frame_context);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }

            _ => {}
        }
    }
}

impl std::ops::Drop for FractalSimulation {
    fn drop(&mut self) {
        self.vks.wait_all_idle();
    }
}

use minmath::{
    colors::RGBAColor,
    mat4::Mat4F32,
    vec2::{normalize, Vec2F32},
};

struct CubicBezier {
    ctrl_points: [Vec2F32; 4],
}

impl CubicBezier {
    fn eval(&self, t: f32) -> Vec2F32 {
        assert!(t >= 0f32 && t <= 1f32);

        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1f32 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        self.ctrl_points[0] * mt3
            + 3f32 * self.ctrl_points[1] * mt2 * t
            + 3f32 * self.ctrl_points[2] * mt * t2
            + t3 * self.ctrl_points[3]
    }

    fn derivative(&self, t: f32) -> Vec2F32 {
        assert!(t >= 0f32 && t <= 1f32);

        let mt = 1f32 - t;
        let a = mt * mt;
        let b = 2f32 * mt * t;
        let c = t * t;

        3f32 * a * (self.ctrl_points[1] - self.ctrl_points[0])
            + 3f32 * b * (self.ctrl_points[2] - self.ctrl_points[1])
            + 3f32 * c * (self.ctrl_points[3] - self.ctrl_points[2])
    }

    fn normal(&self, d: Vec2F32) -> Vec2F32 {
        use minmath::vec2::perp_vec;
        perp_vec(normalize(d))
    }

    fn subdivide_impl(
        &self,
        vertices: &mut Vec<Vec2F32>,
        p0: Vec2F32,
        p1: Vec2F32,
        p2: Vec2F32,
        p3: Vec2F32,
    ) {
        vertices.push(p0);

        //
        // test if curve is straight
        let dst0 = segment_point_dst_square(p0, p3, p1);
        let dst1 = segment_point_dst_square(p0, p3, p2);

        const DST_EPSILON: f32 = 1.0E-6f32;

        if dst0 < DST_EPSILON && dst1 < DST_EPSILON {
            return;
        }

        //
        // subdivide curve
        let l1 = (p0 + p1) * 0.5f32;
        let h = (p1 + p2) * 0.5f32;
        let l2 = (l1 + h) * 0.5f32;
        let r2 = (p2 + p3) * 0.5f32;
        let r1 = (h + r2) * 0.5f32;
        let mid = (l2 + r1) * 0.5f32;

        self.subdivide_impl(vertices, p0, l1, l2, mid);
        self.subdivide_impl(vertices, mid, r1, r2, p3);
    }

    fn subdivide(&self, vertices: &mut Vec<Vec2F32>) {
        self.subdivide_impl(
            vertices,
            self.ctrl_points[0],
            self.ctrl_points[1],
            self.ctrl_points[2],
            self.ctrl_points[3],
        );
    }
}

impl std::convert::From<[(f32, f32); 4]> for CubicBezier {
    fn from(pts: [(f32, f32); 4]) -> Self {
        CubicBezier {
            ctrl_points: [pts[0].into(), pts[1].into(), pts[2].into(), pts[3].into()],
        }
    }
}

impl std::convert::AsRef<[Vec2F32]> for CubicBezier {
    fn as_ref(&self) -> &[Vec2F32] {
        &self.ctrl_points
    }
}

impl std::convert::AsMut<[Vec2F32]> for CubicBezier {
    fn as_mut(&mut self) -> &mut [Vec2F32] {
        &mut self.ctrl_points
    }
}

#[derive(Copy, Clone, Debug)]
struct VertexPC {
    pos: Vec2F32,
    color: RGBAColor,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct UniformCanvas {
    wvp: minmath::mat4::Mat4F32,
}

struct Canvas {
    vertex_buffer: UniqueBuffer,
    index_buffer: UniqueBuffer,
    ubo_vs: UniqueBuffer,
    pipeline: UniquePipeline,
    descriptor_sets: Vec<DescriptorSet>,
    vertices: Vec<VertexPC>,
    indices: Vec<u16>,
}

impl Canvas {
    const MAX_VERTICES: u32 = 8192;
    const MAX_INDICES: u32 = 16535;
    const CIRCLE_TESS_FACTOR: u32 = 32;

    fn new(vks: &mut VulkanState) -> Canvas {
        let vertex_buffer = UniqueBuffer::new::<VertexPC>(
            vks,
            BufferUsageFlags::VERTEX_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            (Self::MAX_VERTICES * vks.swapchain.max_frames) as usize,
        );

        let index_buffer = UniqueBuffer::new::<u16>(
            vks,
            BufferUsageFlags::INDEX_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            (Self::MAX_INDICES * vks.swapchain.max_frames) as usize,
        );

        let ubo_vs = UniqueBuffer::new::<UniformCanvas>(
            vks,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            vks.swapchain.max_frames as usize,
        );

        let pipeline = Self::create_graphics_pipeline(vks);
        let descriptor_sets = unsafe {
            vks.ds.device.allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(vks.ds.descriptor_pool)
                    .set_layouts(&pipeline.descriptor_set_layout),
            )
        }
        .expect("Failed to allocate descriptor sets");

        unsafe {
            vks.ds.device.update_descriptor_sets(
                &[*WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(&[*DescriptorBufferInfo::builder()
                        .buffer(ubo_vs.handle)
                        .offset(0)
                        .range(ubo_vs.item_aligned_size as DeviceSize)])],
                &[],
            );
        }

        Canvas {
            vertex_buffer,
            index_buffer,
            ubo_vs,
            pipeline,
            descriptor_sets,
            vertices: Vec::with_capacity(Self::MAX_VERTICES as usize),
            indices: Vec::with_capacity(Self::MAX_INDICES as usize),
        }
    }

    fn create_graphics_pipeline(vks: &mut VulkanState) -> UniquePipeline {
        let vsm = compile_shader_from_file("data/shaders/canvas.vert", &vks.ds.device).unwrap();
        let fsm = compile_shader_from_file("data/shaders/canvas.frag", &vks.ds.device).unwrap();

        let pipeline_create_info = *GraphicsPipelineCreateInfo::builder()
            .input_assembly_state(
                &PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(PrimitiveTopology::TRIANGLE_LIST),
            )
            .stages(&[
                *PipelineShaderStageCreateInfo::builder()
                    .module(*vsm)
                    .stage(ShaderStageFlags::VERTEX)
                    .name(&CStr::from_bytes_with_nul(b"main\0").unwrap()),
                *PipelineShaderStageCreateInfo::builder()
                    .module(*fsm)
                    .stage(ShaderStageFlags::FRAGMENT)
                    .name(&CStr::from_bytes_with_nul(b"main\0").unwrap()),
            ])
            .vertex_input_state(
                &PipelineVertexInputStateCreateInfo::builder()
                    .vertex_attribute_descriptions(&[
                        *VertexInputAttributeDescription::builder()
                            .binding(0)
                            .format(ash::vk::Format::R32G32_SFLOAT)
                            .location(0)
                            .offset(0),
                        *VertexInputAttributeDescription::builder()
                            .binding(0)
                            .format(ash::vk::Format::R8G8B8A8_UNORM)
                            .location(1)
                            .offset(8),
                    ])
                    .vertex_binding_descriptions(&[*VertexInputBindingDescription::builder()
                        .binding(0)
                        .input_rate(VertexInputRate::VERTEX)
                        .stride(size_of::<VertexPC>() as u32)]),
            )
            .viewport_state(
                &PipelineViewportStateCreateInfo::builder()
                    .viewports(&[*Viewport::builder()
                        .x(0f32)
                        .y(0f32)
                        .width(vks.ds.surface.image_size.width as f32)
                        .height(vks.ds.surface.image_size.height as f32)
                        .min_depth(0f32)
                        .max_depth(1f32)])
                    .scissors(&[Rect2D {
                        offset: Offset2D { x: 0, y: 0 },
                        extent: Extent2D {
                            width: vks.ds.surface.image_size.width,
                            height: vks.ds.surface.image_size.height,
                        },
                    }]),
            )
            .rasterization_state(
                &PipelineRasterizationStateCreateInfo::builder()
                    .cull_mode(CullModeFlags::NONE)
                    .front_face(FrontFace::COUNTER_CLOCKWISE)
                    .polygon_mode(PolygonMode::FILL)
                    .line_width(1f32),
            )
            .multisample_state(
                &PipelineMultisampleStateCreateInfo::builder()
                    .rasterization_samples(SampleCountFlags::TYPE_1),
            )
            .color_blend_state(
                &PipelineColorBlendStateCreateInfo::builder().attachments(&[
                    *PipelineColorBlendAttachmentState::builder()
                        .color_write_mask(ColorComponentFlags::RGBA)
                        .blend_enable(false)
                        .alpha_blend_op(BlendOp::ADD)
                        .color_blend_op(BlendOp::ADD)
                        .src_color_blend_factor(BlendFactor::SRC_ALPHA)
                        .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                        .src_alpha_blend_factor(BlendFactor::ONE)
                        .dst_alpha_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA),
                ]),
            )
            .dynamic_state(
                &PipelineDynamicStateCreateInfo::builder()
                    .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR]),
            )
            .render_pass(vks.renderpass)
            .subpass(0);

        UniquePipeline::new(
            vks,
            &[*DescriptorSetLayoutCreateInfo::builder().bindings(&[
                *DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .stage_flags(ShaderStageFlags::VERTEX),
            ])],
            pipeline_create_info,
        )
    }

    pub fn add_circle(&mut self, radius: f32, origin: Vec2F32, color: RGBAColor) {
        let angle_step = 2f32 * std::f32::consts::PI / Self::CIRCLE_TESS_FACTOR as f32;

        let idx_offset = self.vertices.len();

        self.vertices.push(VertexPC { pos: origin, color });

        for i in 1..Self::CIRCLE_TESS_FACTOR + 2 {
            let pos = origin
                + Vec2F32 {
                    x: radius * ((i - 1) as f32 * angle_step).cos(),
                    y: radius * ((i - 1) as f32 * angle_step).sin(),
                };

            self.vertices.push(VertexPC { pos, color });
        }

        (0..Self::CIRCLE_TESS_FACTOR as usize).for_each(|idx| {
            self.indices.push((idx_offset + 0) as u16);
            self.indices.push((idx_offset + idx + 1) as u16);
            self.indices.push((idx_offset + idx + 2) as u16);
        });
    }

    pub fn add_double_circle(
        &mut self,
        inner_radius: f32,
        outer_radius: f32,
        origin: Vec2F32,
        inner_color: RGBAColor,
        outer_color: RGBAColor,
    ) {
        self.add_circle(outer_radius, origin, outer_color);
        self.add_circle(inner_radius, origin, inner_color);
    }

    pub fn add_line(
        &mut self,
        start: Vec2F32,
        end: Vec2F32,
        width: f32,
        color0: RGBAColor,
        color1: RGBAColor,
    ) {
        let dir = normalize(minmath::vec2::perp_vec(end - start));
        let idx_offset = self.vertices.len() as u16;

        self.vertices.extend(
            [
                VertexPC {
                    pos: start - dir * width * 0.5f32,
                    color: color0,
                },
                VertexPC {
                    pos: end - dir * width * 0.5f32,
                    color: color1,
                },
                VertexPC {
                    pos: end + dir * width * 0.5f32,
                    color: color1,
                },
                VertexPC {
                    pos: start + dir * width * 0.5f32,
                    color: color0,
                },
            ]
            .iter(),
        );

        self.indices
            .extend([0, 1, 2, 0, 2, 3].iter().map(|&i| idx_offset + i as u16));
    }

    pub fn add_rectangle(&mut self, org: Vec2F32, width: f32, height: f32, color: RGBAColor) {
        let idx_offset = self.vertices.len() as u16;

        self.vertices.extend(
            [
                VertexPC {
                    pos: Vec2F32 {
                        x: org.x,
                        y: org.y + height,
                    },
                    color,
                },
                VertexPC {
                    pos: Vec2F32 {
                        x: org.x + width,
                        y: org.y + height,
                    },
                    color,
                },
                VertexPC {
                    pos: Vec2F32 {
                        x: org.x + width,
                        y: org.y,
                    },
                    color,
                },
                VertexPC { pos: org, color },
            ]
            .iter(),
        );

        self.indices
            .extend([0, 1, 2, 0, 2, 3].iter().map(|&i| idx_offset + i as u16));
    }

    pub fn draw(&mut self, vks: &VulkanState, frame_context: &FrameRenderContext) {
        UniqueBufferMapping::new(
            &self.vertex_buffer,
            &vks.ds,
            Some(
                self.vertex_buffer.item_aligned_size
                    * Self::MAX_VERTICES as usize
                    * frame_context.current_frame_id as usize,
            ),
            Some(Self::MAX_VERTICES as usize * self.vertex_buffer.item_aligned_size),
        )
        .write_data(&self.vertices);

        UniqueBufferMapping::new(
            &self.index_buffer,
            &vks.ds,
            Some(
                self.index_buffer.item_aligned_size
                    * Self::MAX_INDICES as usize
                    * frame_context.current_frame_id as usize,
            ),
            Some(Self::MAX_INDICES as usize * self.index_buffer.item_aligned_size),
        )
        .write_data(&self.indices);

        UniqueBufferMapping::new(
            &self.ubo_vs,
            &vks.ds,
            Some(self.ubo_vs.item_aligned_size * frame_context.current_frame_id as usize),
            Some(size_of::<UniformCanvas>()),
        )
        .write_data(&[UniformCanvas {
            wvp: minmath::projection::vk::ortho_symmetric(
                frame_context.fb_size.width as f32,
                frame_context.fb_size.height as f32,
                0.001f32,
                1000f32,
            ),
        }]);

        unsafe {
            vks.ds.device.cmd_set_viewport(
                frame_context.cmd_buff,
                0,
                &[Viewport {
                    x: 0f32,
                    y: 0f32,
                    width: frame_context.fb_size.width as f32,
                    height: frame_context.fb_size.height as f32,
                    min_depth: 1f32,
                    max_depth: 0f32,
                }],
            );
            vks.ds.device.cmd_set_scissor(
                frame_context.cmd_buff,
                0,
                &[Rect2D {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: frame_context.fb_size,
                }],
            );

            vks.ds.device.cmd_bind_vertex_buffers(
                frame_context.cmd_buff,
                0,
                &[self.vertex_buffer.handle],
                &[self.vertex_buffer.item_aligned_size as DeviceSize
                    * (Self::MAX_VERTICES * frame_context.current_frame_id) as DeviceSize],
            );
            vks.ds.device.cmd_bind_index_buffer(
                frame_context.cmd_buff,
                self.index_buffer.handle,
                self.index_buffer.item_aligned_size as DeviceSize
                    * (Self::MAX_INDICES * frame_context.current_frame_id) as DeviceSize,
                IndexType::UINT16,
            );

            vks.ds.device.cmd_bind_pipeline(
                frame_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.handle,
            );

            vks.ds.device.cmd_bind_descriptor_sets(
                frame_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &self.descriptor_sets,
                &[self.ubo_vs.item_aligned_size as u32 * frame_context.current_frame_id],
            );

            vks.ds.device.cmd_draw_indexed(
                frame_context.cmd_buff,
                self.indices.len() as u32,
                1,
                0,
                0,
                0,
            );
        }

        self.indices.clear();
        self.vertices.clear();
    }
}

fn segment_point_dst_square(s0: Vec2F32, s1: Vec2F32, p: Vec2F32) -> f32 {
    use minmath::vec2::dot;

    let w = p - s0;
    let dir = s1 - s0;
    let proj = dot(w, dir);

    if proj <= 0f32 {
        return dot(w, w);
    }

    let vsq = dot(dir, dir);
    if proj >= vsq {
        return dot(w, w) - 2f32 * proj + vsq;
    }

    dot(w, w) - (proj / vsq) * proj
}
