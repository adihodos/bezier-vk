#version 460 core

in VS_OUT_FS_IN {
    layout (location = 0) vec4 color;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
    FinalFragColor = fs_in.color;
}
