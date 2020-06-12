#version 450

layout (location = 0) in float inColor;

layout (location = 0) out float outFragColor;

void main() 
{
  outFragColor = inColor;
}