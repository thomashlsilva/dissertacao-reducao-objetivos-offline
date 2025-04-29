## 📋 Resumo da Pesquisa  

Este trabalho propõe uma abordagem **híbrida (MOEA + ML)** para otimização multiobjetivo de alta dimensionalidade (MaOPs), combinando:  
- **Redução de dimensionalidade** via:  
  - *Seleção de atributos*: Laplacian Score (LS).  
  - *Extração de atributos*: Análise de Componentes Principais (PCA).  
- **Otimização** com NSGA-III (estado da arte).  

### 🔬 Principais Resultados  
✅ **PCA** superou LS em problemas de **alta dimensionalidade** (família DTLZ).  
✅ Técnicas permitiram uso de MOEAs tradicionais em MaOPs, com ganhos em:  
  - Escalabilidade.  
  - Qualidade das soluções (avaliada por `IGD⁺` e `Iε⁺`).  
  - Visualização e tomada de decisão.  
⚠️ **Limitação**: Abordagens *offline* exigem robustez adicional (sugerido para trabalhos futuros).  

### 📊 Métricas e Validação  
- Testes estatísticos: **Teste de Postos Sinalizados de Wilcoxon** (confiança de 95%).  
- Indicadores: `IGD⁺`, `Iε⁺` + análise de *boxplots*.  

**Palavras-chave**: MaOPs, Redução de dimensionalidade, Laplacian Score, PCA, NSGA-III, DTLZ.  

---

## 📁 Dados e Resultados  
- [Fonte de dados para as funções objetivo](https://drive.google.com/file/d/1LhPU-UBnTzN96sGIKPeUwqhGmxalrzEQ/view?usp=sharing)  
- [Resultados Laplacian Score](https://drive.google.com/file/d/1DBsy9ZHunAmDOnN4iPqD_0ahEc2gN5lj/view?usp=sharing)  
- [Resultados PCA](https://drive.google.com/file/d/1Y54pvJhc8LEBp_PjqrGWxGJiM7OlbXUU/view?usp=sharing)  

📌 **Detalhes de parametrização**: Consulte o capítulo de *Experimentos Computacionais* no texto.  

---

## 📚 Como Citar  
```bibtex  
@mastersthesis{silva2024,
  author  = {Thomás Henrique Lopes Silva},
  title   = {Redução de objetivos em MaOPs por meio de aprendizado de máquina não supervisionado: 
             Uma abordagem com seleção e extração de atributos},
  school  = {Centro Federal de Educação Tecnológica de Minas Gerais},
  year    = {2024},
  address = {Belo Horizonte, Brasil},
  url     = {https://github.com/thomashlsilva/dissertacao-reducao-objetivos-offline},
  note    = {Código e dados disponíveis em: \url{https://github.com/thomashlsilva/dissertacao-reducao-objetivos-offline}}
}
```  

**Contato**:  
✉️ [thomashlsilva@gmail.com](mailto:thomashlsilva@gmail.com)  
💻 [@thomashlsilva](https://github.com/thomashlsilva)  

---

## 📜 Licença  
- Códigos: [![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
- Dados/Resultados: [![CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](LICENSE-CC-BY-4.0)