#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// Representa uma cidade
typedef struct{
    float x;
    float y;
}Cidade;

// Representa uma rota
typedef struct{
    Cidade **rota;
    int tamanho;
    float fitness;
}Rota;

// Representa uma lista de cidades
typedef struct{
    Cidade **cidades;
    int tamanho;
}ListaCidades;

double random_float(float max); // Retorna um float random
float distancia(Cidade *a, Cidade *b); // Calcula a distância entre 2 cidades
float fitness(Cidade **cidades, int size); // Calcula a aptidão de cada rota
Cidade *cria_cidade(void); // Cria uma cidade aleatoria
ListaCidades *cria_lista_cidades(int qt); // Cria uma lista de cidades de tamanho qt
int check_index(int *indices, int qt, int index); // Usado para verificar se o indice já foi usado


int main(void){
    ListaCidades *a = NULL;

    a = cria_lista_cidades(20);

    for(int i = 0; i < a->tamanho; ++i){
        printf("%d -> (%f,%f)\n", i, a->cidades[i]->x, a->cidades[i]->y);
    }

    Rota *rotas = NULL;
    rotas = malloc( * sizeof(Rota))

    return 0;
}

// Gera um float random!
double random_float(float max){
    double x;

    x = ((double) rand()/(double)(RAND_MAX)) * max;

    return x;
}

// Calcula a distância entre 2 cidades
float distancia(Cidade *a, Cidade *b){

    float dis_x = pow((a->x - b->x), 2); // distancia do eixo x
    float dis_y = pow((a->y - b->y), 2); // distancia do eixo y

    // Retorna a distância euclideana entre os 2
    return sqrt(dis_x + dis_y);
}//distancia

// Calcula a aptidão de cada rota
float fitness(Cidade **cidades, int size){
    // Distancia total das rotas
    float distancia_total = 0.0;

    // Usado para ter a posição de cada cidade
    Cidade *atual = NULL;
    Cidade *seguinte = NULL;

    // Percorre a sequencia de cidades
    for(int i = 0; i < size; ++i){

        // Cidade Atual
        atual = (Cidade *) cidades[i];

        // Pega a próxima cidade
        if(i < size -1){
            seguinte = cidades[i+1];
        }
        else{
            seguinte = cidades[0];
        }

        // Adiciona a distancia entre as cidades para a distância total
        distancia_total += distancia(atual, seguinte);
    }

    // Retorna a distância total
    return distancia_total;

}//fitness

// Cria uma cidade com posição aleatória
Cidade *cria_cidade(void){
    Cidade *nova = NULL;

    // Cria uma cidade com localização aleatória
    nova = malloc(sizeof(Cidade));
    nova->x = random_float(200);
    nova->y = random_float(200);

    return nova;
}//cria_cidade

ListaCidades *cria_lista_cidades(int qt){
    Cidade **cidades = NULL;
    ListaCidades *lista = NULL;
    lista = malloc(sizeof(ListaCidades));
    cidades = malloc(qt * sizeof(Cidade *));

    // Adiciona cidades aleatórias
    for(int i = 0; i < qt; ++i){
        cidades[i] = (Cidade *) cria_cidade();
    }

    // Adiciona a lista de cidades
    lista->cidades = cidades;
    lista->tamanho = qt;

    return lista;
}//cria_lista_cidades


int check_index(int *indices, int qt, int index){
    for(int i = 0; i < qt; ++i){
        if(indices[i] == index)
            return 1;
    }

    return 0;
}

ListaCidades *cria_rota(ListaCidades *a){
    Cidade **rota = NULL;
    int *indices = NULL;

    rota = malloc(a->tamanho * sizeof(rota *));
    indices = malloc(a->tamanho * sizeof(int));
    for(int i = 0; i < a->tamanho; ++i){
        indices[i] = -1;
    }

    int index;
    for(int i = 0; i < a->tamanho; ++i){
        index = rand() % a->tamanho;

        while(check_index(indices, a->tamanho, index)){
            index += 1;
            if(index >= a->tamanho)
                index = 0;
        }

        indices[i] = index;
        rota[i] = a->cidades[index];
    }

    ListaCidades *nova = NULL;
    nova = malloc(sizeof(ListaCidades));
    nova->cidades = rota;
    nova->tamanho = a->tamanho;

    return nova;
}//cria_rota