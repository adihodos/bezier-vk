#version 460 core

layout (location = 0) in vec2 vs_in_pos;
layout (location = 1) in vec4 vs_in_color;

layout (set = 0, binding = 0, row_major) uniform Transforms {
    mat4 wvp;
} tf;

out VS_OUT_FS_IN {
    layout (location = 0) vec4 color;
} vs_out;

void main() {
    gl_Position = tf.wvp * vec4(vs_in_pos, 1, 1.0);
    vs_out.color = vs_in_color;
}
