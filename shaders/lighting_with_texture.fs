#version 330 core
out vec4 FragColor;

struct Material {
    vec3 diffuse;
    vec3 specular;    
    float shininess;
}; 

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;  
in vec3 Normal;  
in vec2 TexCoords;
  
uniform vec3 viewPos;
uniform Material material;
uniform Light light;
//texture
uniform sampler2D texture_diffuse1;

void main()
{
    
    // ambient
    vec4 textel = texture(texture_diffuse1, TexCoords);

    vec3 ambient = light.ambient * vec3(textel);
  	
    float frontFace = 1.0;
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    if (diff < 1e-12){
        frontFace = 0.8;
    }else{
        frontFace = 1.0;
    }
    //vec3 diffuse = light.diffuse * diff * texture(material.diffuse, TexCoords).rgb;  
    vec3 diffuse = light.diffuse * diff * vec3(textel);

    // specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * material.specular;  
        
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result * frontFace, 1.0);    
} 