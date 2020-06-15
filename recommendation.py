import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import missingno as msno
import ppscore as pps
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO
pd.set_option('display.max_columns', 500)



@st.cache
def readcsv(csv):
    df=pd.read_csv(csv)
    return df

def load_data():
    DATA_URL = ('https://raw.githubusercontent.com/guireis1/Codenation-Final-Project/master/estaticos_portfolio1.csv')
    data = pd.read_csv(DATA_URL)
    return data

def head(dataframe):
    if len(dataframe) > 1000:
        lenght = 1000
    else:
        lenght = len(dataframe)
    slider = st.slider('Linhas exibidas:', 10, lenght)
    st.dataframe(dataframe.head(slider))

def vis(data):
    for i in data.columns:
        if i == 'setor':
            sns.set(style="whitegrid")
            plt.figure(figsize=(20,10))
            sns.countplot(x="setor", data=data, palette="Reds_r",saturation=0.5)
            plt.title('Contagem dos Setores',fontsize=20)
            plt.xlabel('')
            plt.ylabel('')
            st.pyplot()
        if i == 'natureza_juridica_macro':
            sns.set(style="whitegrid")
            plt.figure(figsize=(20,10))
            sns.countplot(x="natureza_juridica_macro", data=data, palette="Reds_r",saturation=0.5)
            plt.title('Contagem da natureza jurídica',fontsize=20)
            plt.xlabel('')
            plt.ylabel('')
            st.pyplot()       
        if i == 'de_faixa_faturamento_estimado_grupo':
            sns.set(style="whitegrid")
            plt.figure(figsize=(20,20))
            sns.countplot(y="de_faixa_faturamento_estimado_grupo",hue='setor', data=data, palette="Reds_r",saturation=0.5)
            plt.title('Contagem do faturamento por setor',fontsize=20)
            plt.xlabel('')
            plt.ylabel('')
            st.pyplot()
        if i == 'nm_meso_regiao':
            sns.set(style="whitegrid")
            plt.figure(figsize=(20,20))
            sns.countplot(y="nm_meso_regiao", data=data, palette="Reds_r",saturation=0.5)
            plt.title('Contagem Meso Região',fontsize=20)
            plt.xlabel('')
            plt.ylabel('')
            st.pyplot()
@st.cache
def descritiva(dataframe):
    desc = dataframe.describe().T
    desc['column'] = desc.index
    exploratory = pd.DataFrame()
    exploratory['NaN'] = dataframe.isnull().sum().values
    exploratory['NaN %'] = 100 * (dataframe.isnull().sum().values / len(dataframe))
    exploratory['NaN %'] = exploratory['NaN %'].apply(lambda x: str(round(x,2)) + " %")
    exploratory['column'] = dataframe.columns
    exploratory['dtype'] = dataframe.dtypes.values
    exploratory = exploratory.merge(desc, on='column', how='left')
    exploratory.loc[exploratory['dtype'] == 'object', 'count'] = len(dataframe) - exploratory['NaN']
    exploratory.set_index('column', inplace=True)
    return exploratory

def missing(data,nome):
    #plt.figure(figsize=(10,15))
    msno.bar(data,sort='descending')
    plt.title(nome,fontsize=30)
    st.pyplot()

def missing_dendo(data,nome):
    msno.dendrogram(data)
    plt.title(nome,fontsize=30)
    st.pyplot()

def geoloc(data,coord):
    coordenadas = []
    null_count= 0

    for local in data['nm_micro_regiao']:
        
        coords=coord[coord['nome']==local][['latitude','longitude']]
        
        if not coords.empty:
            
            coordenadas.append([coords['latitude'].values[0]-random.uniform(0,0.25),
                                coords['longitude'].values[0]-random.uniform(0,0.25)])
        else:
            null_count += 1
    print(null_count)

    return coordenadas
  #  st.map(coordenadas)

#for mark in coordenadas:
 #   folium.Marker([mark[0],mark[1]],icon=folium.Icon(icon='exclamation',color='darkred',prefix='fa')).add_to(m)
  #  print(null_count_port)
def recommend(port_pre,slider_nn,market_col_select_scaled,market):

    valor_nn = slider_nn
    nn= NearestNeighbors(n_neighbors=valor_nn,metric='cosine')
    nn.fit(market_col_select_scaled)

    nn_port_list = {}
    
    for row in range(port_pre.shape[0]):
        nn_port_list[row] = nn.kneighbors(port_pre.iloc[[row]].values)
        
    nn_size = len(nn_port_list)
    nn_num = len(nn_port_list[0][1][0])
    
    nn_index = nn_port_list[0][1][0]
    nn_distance = nn_port_list[0][0][0]
    
    np.delete(nn_index, [0,1])
    np.delete(nn_distance, [0,1])
    
    for i in range(1,nn_size):
        nn_index = np.concatenate((nn_index,nn_port_list[i][1][0]),axis=None)
        nn_distance = np.concatenate((nn_distance,nn_port_list[i][0][0]),axis=None)
        
    if len(nn_index) != nn_size*nn_num:
        print ('Erro')
    
    id_origin = {}
    
    for idx,ind in zip(nn_index,range(len(nn_index))):
        id_origin[ind] = (port_pre.iloc[int(ind/valor_nn)].name , market.iloc[idx].name, (nn_distance[ind])) 
        
    recommend = pd.DataFrame.from_dict(id_origin,orient='index')
    recommend.rename(columns={0:'id_origin',1:'id',2:'distance'},inplace=True)
    recommend=recommend[recommend['id'].isin(port_pre.index)==0]  #tirando os conflitos
    recommend.set_index('id',inplace=True)
    suggestion = recommend.merge(market, how='left', left_index=True,right_index=True) ##unindo com o market
    suggestion = suggestion.loc[~suggestion.index.duplicated(keep='first')] ##tirando os duplicados   

    return suggestion

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="recomendacao.csv">Download csv file</a>' 
  #   

def main():
    
    st.image('img/gui_logo.jpeg', use_column_width=True)
    st.header('Bem vindo!')
    st.subheader('**Você está no sistema de recomendação de clientes**')
    st.markdown('O sistema recomendará novos clientes baseado em comparações com os seus atuais clientes de forma customizada a partir das características desejadas.')

    st.markdown('###  Precisamos que você nos forneça o **portifólio de seus clientes!**')
    st.markdown(' *Obs.: Caso você não tenha um portifólio para usar, escolha um [desses](https://github.com/guireis1/Codenation-Final-Project/tree/master/data). *')
    file3= st.file_uploader('Upload clientes.csv',type='csv')

    if file3 is not None:

        
        market_pre = pd.read_csv('data/data_preprocess.csv')
        market = pd.read_csv('data/market.csv')
        #market = pd.DataFrame(readcsv(file2))
        #market= pd.read_csv(file2)
        #market_pre = pd.DataFrame(readcsv(file1))
        #market_pre = pd.read_csv(file1)
        port = pd.DataFrame(readcsv(file3))  
        st.text('Loading data...done!')


        #Començando o processamento
        #market = pd.read_csv('market.csv')
        #market_pre = pd.read_csv('data_preprocess.csv')
        #port = pd.read_csv('data/estaticos_portfolio1.csv')  

        market_pre.set_index('id',inplace=True)
        market.set_index(market_pre.index,inplace=True)
        market.drop('Unnamed: 0',axis=1,inplace=True)

        port= port.set_index('id')
        port.drop(port.columns,axis=1,inplace=True) 

        port_market = market.merge(port,how='right',left_index=True,right_index=True)
        port_market_pre = market_pre.merge(port,how='right',left_index=True,right_index=True)

        st.markdown('DataFrame do Portofólio:')
        head(port_market)
        #Todos datasets prontos
        #st.sidebar.image(st.image('img/logo.png', use_column_width=True))
        st.sidebar.header('Opções de análise do Portifólio:')
        sidemulti = st.sidebar.multiselect('Escolha: ',('Visualização','Descritiva','Geolocalização'))
        
        if ('Visualização' in sidemulti):
            st.markdown('## **Visualização do Portifólio**')
            st.markdown('Perfil de clientes considerando features importantes')
            vis(port_market)
            st.markdown('*Para melhor visualização clique na imagem*')
        if ('Descritiva' in sidemulti):
            st.markdown('## **Análise Descritiva do Portifólio**')
            st.dataframe(descritiva(port_market))     
            missing(port_market,'Visualização dos nulos do Portifólio')
            missing_dendo(port_market,'Dendograma dos nulos do Portifólio') 
            st.markdown('*Para melhor visualização clique na imagem*')
        if ('Geolocalização' in sidemulti):
            coordenadas = pd.read_csv('https://raw.githubusercontent.com/guireis1/Codenation-Final-Project/master/data/coordenadas')
            coordenadas.drop('Unnamed: 0',axis=1,inplace=True)
            st.markdown('## **Geolocalização do Portifólio**')
            st.markdown('Localização das empresas contidas no portifólio')
            cord_port = geoloc(port_market,coordenadas)
            cord_port_df=pd.DataFrame(cord_port,columns=('lat','lon'))
            st.map(cord_port_df)

        st.sidebar.header('Opções de análise do mercado:')
        sidemulti_market = st.sidebar.multiselect('Escolha: ',('Visualização','Descritiva','Correlação','Análise dos Nulos','Colunas excluídas'))

        if ('Visualização' in sidemulti_market):
            st.markdown('## **Visualização do Mercado**')
            vis(market)
            st.markdown('*Para melhor visualização clique na imagem*')
        if ('Descritiva' in sidemulti_market):
            st.markdown('## **Análise Descritiva do Mercado**')
            st.dataframe(descritiva(market))     
            #missing(market,'Visualização dos nulos')
            #missing_dendo(market,'Dendograma nulos')

        if ('Correlação' in sidemulti_market):
            st.markdown('## **Correlações do Mercado**')
            st.markdown('Correlação padrão')
            st.image('img/corr_matrix.png', use_column_width=True)
            st.markdown('Correlação usando PPS')
            st.image('img/corr_pps.png', use_column_width=True)

        if ('Análise dos Nulos' in sidemulti_market):
            st.markdown('## **Análise dos nulos **')

            st.markdown('### **Colunas Numéricas:**')
            st.image('img/valores20.png', use_column_width=True)
            st.image('img/valores60.png', use_column_width=True)
            st.image('img/valores80.png', use_column_width=True)
            st.image('img/dendo_90.png', use_column_width=True)
            st.image('img/dendo100.png', use_column_width=True)

            st.markdown('### **Colunas Categoricas:**')
            st.image('img/valores_nulos.png', use_column_width=True)
            st.image('img/dendo_cat.png', use_column_width=True)
        if ('Colunas excluídas' in sidemulti_market):
            col_excluidas=[ 'sg_uf', 'idade_emp_cat', 'fl_me', 'fl_sa', 'fl_epp', 'fl_ltda', 'dt_situacao', 'fl_st_especial', 'nm_divisao', 'nm_segmento', 'fl_spa',
 'vl_total_tancagem', 'vl_total_veiculos_antt', 'fl_optante_simples', 'qt_art', 'vl_total_veiculos_pesados_grupo', 'vl_total_veiculos_leves_grupo', 'vl_total_tancagem_grupo',
 'vl_total_veiculos_antt_grupo', 'vl_potenc_cons_oleo_gas', 'fl_optante_simei', 'sg_uf_matriz', 'de_saude_rescencia', 'nu_meses_rescencia', 'de_indicador_telefone',
 'fl_simples_irregular', 'vl_frota', 'qt_socios_pf', 'qt_socios_pj', 'idade_maxima_socios', 'idade_minima_socios', 'qt_socios_st_regular', 'qt_socios_st_suspensa',
 'qt_socios_masculino', 'qt_socios_feminino', 'qt_socios_pep', 'qt_alteracao_socio_total', 'qt_alteracao_socio_90d', 'qt_alteracao_socio_180d', 'qt_alteracao_socio_365d',
 'qt_socios_pj_ativos', 'qt_socios_pj_nulos', 'qt_socios_pj_baixados', 'qt_socios_pj_suspensos', 'qt_socios_pj_inaptos', 'vl_idade_media_socios_pj', 'vl_idade_maxima_socios_pj',
 'vl_idade_minima_socios_pj', 'qt_coligados', 'qt_socios_coligados', 'qt_coligados_matriz', 'qt_coligados_ativo', 'qt_coligados_baixada', 'qt_coligados_inapta',
 'qt_coligados_suspensa', 'qt_coligados_nula', 'idade_media_coligadas', 'idade_maxima_coligadas', 'idade_minima_coligadas', 'coligada_mais_nova_ativa',
 'coligada_mais_antiga_ativa', 'idade_media_coligadas_ativas', 'coligada_mais_nova_baixada', 'coligada_mais_antiga_baixada', 'idade_media_coligadas_baixadas',
 'qt_coligados_sa', 'qt_coligados_me', 'qt_coligados_mei', 'qt_coligados_ltda', 'qt_coligados_epp', 'qt_coligados_norte', 'qt_coligados_sul', 'qt_coligados_nordeste',
 'qt_coligados_centro', 'qt_coligados_sudeste', 'qt_coligados_exterior', 'qt_ufs_coligados', 'qt_regioes_coligados', 'qt_ramos_coligados', 'qt_coligados_industria',
 'qt_coligados_agropecuaria', 'qt_coligados_comercio', 'qt_coligados_serviço', 'qt_coligados_ccivil', 'qt_funcionarios_coligados',
 'qt_funcionarios_coligados_gp', 'media_funcionarios_coligados_gp', 'max_funcionarios_coligados_gp', 'min_funcionarios_coligados_gp', 'vl_folha_coligados', 'media_vl_folha_coligados', 'max_vl_folha_coligados', 'min_vl_folha_coligados', 'vl_folha_coligados_gp', 'media_vl_folha_coligados_gp',
 'max_vl_folha_coligados_gp', 'min_vl_folha_coligados_gp', 'faturamento_est_coligados', 'media_faturamento_est_coligados', 'max_faturamento_est_coligados', 'min_faturamento_est_coligados',
 'faturamento_est_coligados_gp', 'media_faturamento_est_coligados_gp', 'max_faturamento_est_coligados_gp', 'min_faturamento_est_coligados_gp', 'total_filiais_coligados', 'media_filiais_coligados', 'max_filiais_coligados',
 'min_filiais_coligados', 'qt_coligados_atividade_alto', 'qt_coligados_atividade_medio', 'qt_coligados_atividade_baixo', 'qt_coligados_atividade_mt_baixo', 'qt_coligados_atividade_inativo',
 'qt_coligadas', 'sum_faturamento_estimado_coligadas', 'de_faixa_faturamento_estimado', 'vl_faturamento_estimado_aux', 'vl_faturamento_estimado_grupo_aux', 'qt_ex_funcionarios',
 'qt_funcionarios_grupo', 'percent_func_genero_masc', 'percent_func_genero_fem', 'idade_ate_18', 'idade_de_19_a_23', 'idade_de_24_a_28', 'idade_de_29_a_33',
 'idade_de_34_a_38', 'idade_de_39_a_43', 'idade_de_44_a_48', 'idade_de_49_a_53', 'idade_de_54_a_58', 'idade_acima_de_58', 'grau_instrucao_macro_analfabeto',
 'grau_instrucao_macro_escolaridade_fundamental', 'grau_instrucao_macro_escolaridade_media', 'grau_instrucao_macro_escolaridade_superior', 'grau_instrucao_macro_desconhecido',
 'total', 'meses_ultima_contratacaco', 'qt_admitidos_12meses', 'qt_desligados_12meses', 'qt_desligados', 'qt_admitidos', 'media_meses_servicos_all', 'max_meses_servicos_all', 'min_meses_servicos_all',
 'media_meses_servicos', 'max_meses_servicos', 'min_meses_servicos', 'qt_funcionarios_12meses', 'qt_funcionarios_24meses', 'tx_crescimento_12meses', 'tx_crescimento_24meses']
            
            st.markdown('## **Colunas excluídas**')
            st.markdown('Decidimos não utiliza-las por quantidade de linhas não preenchidas, grandes correlações com outrar variáveis, pouca importância para o modelo ou redundância!')
            st.markdown('**São elas:**')
            st.write(col_excluidas)

        st.sidebar.header('Sistema de recomendação')
        start_model = st.sidebar.checkbox('Aperte para começarmos a modelagem do sistema!')

        st.sidebar.markdown('**Desenvolvido por,**')
        st.sidebar.markdown('*Guilherme Reis Mendes*')
        st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/guilherme-reis-2862ab153/)')
        st.sidebar.markdown('[GitHub](https://github.com/guireis1/)')
        
        if start_model:
            st.header('**Modelagem**')
            st.subheader('**Primeiro selecione as features que gostaria de usar**')
            st.markdown('*Essas serão as colunas que serão utilizadas no sistema de recomendação!*')
            st.markdown('**Colunas que recomendamos:**')

            col_select=[]

            ramo = st.checkbox('de_ramo')
            idade = st.checkbox('idade_emp_cat')
            meso = st.checkbox('nm_meso_regiao')
            juridica = st.checkbox('natureza_juridica_macro')
            faturamento = st.checkbox('de_faixa_faturamento_estimado_grupo')
            filiais = st.checkbox('qt_filiais')
            mei = st.checkbox('fl_mei')
            rm = st.checkbox('fl_rm')
            
            st.markdown('**Colunas opcionais:**')
            
            setor = st.checkbox('setor')
            rotatividade = st.checkbox('tx_rotatividade')
            idade_socios = st.checkbox('idade_media_socios')
            socios = st.checkbox('qt_socios')
            renda = st.checkbox('empsetorcensitariofaixarendapopulacao')
            leve = st.checkbox('vl_total_veiculos_leves_grupo')
            pesado = st.checkbox('vl_total_veiculos_pesados_grupo')
            iss = st.checkbox('fl_passivel_iss')
            atividade = st.checkbox('de_nivel_atividade')
            saude = st.checkbox('de_saude_tributaria')
            veiculo = st.checkbox('fl_veiculo')
            antt = st.checkbox('fl_antt')
            telefone = st.checkbox('fl_telefone')
            email = st.checkbox('fl_email')
            matriz = st.checkbox('fl_matriz')
            if ramo:
                col_select.append('de_ramo')
            if idade:
                col_select.append('idade_emp_cat')
            if meso:
                col_select.append('nm_meso_regiao')
                meso_ohe=pd.get_dummies(market_pre['nm_meso_regiao'],drop_first=True)
            if faturamento:
                col_select.append('de_faixa_faturamento_estimado_grupo')
            if juridica:
                col_select.append('natureza_juridica_macro')
                juridico_ohe=pd.get_dummies(market_pre['natureza_juridica_macro'],drop_first=True)
            if filiais:
                col_select.append('qt_filiais')
            if mei:
                col_select.append('fl_mei')
            if rm:
                col_select.append('fl_rm') 
            if setor:
                col_select.append('setor')
                setor_ohe=pd.get_dummies(market_pre['setor'],drop_first=True)
            if rotatividade:
                col_select.append('tx_rotatividade')      
            if idade_socios:
                col_select.append('idade_media_socios')          
            if socios:
                col_select.append('qt_socios')    
            if renda:
                col_select.append('empsetorcensitariofaixarendapopulacao')        
            if leve:
                col_select.append('vl_total_veiculos_leves_grupo')    
            if pesado:
                col_select.append('vl_total_veiculos_pesados_grupo')           
            if iss:
                col_select.append('fl_passivel_iss')       
            if atividade:
                col_select.append('de_nivel_atividade')
            if saude:
                col_select.append('de_saude_tributaria')
            if veiculo:
                col_select.append('fl_veiculo')
            if antt:
                col_select.append('fl_antt')
            if telefone:
                col_select.append('fl_telefone')
            if email:
                col_select.append('fl_email')
            if matriz:
                col_select.append('fl_matriz')


            
            st.markdown('## **Podemos continuar?**')
            features_select = st.checkbox('Sim')
            
            
            if features_select:
                st.text('*Colunas selecionadas com sucesso!*')

                st.write('Colunas Selecionadas:' , col_select)

                st.subheader('Agora escolha a quantidade de recomendações que deseja!')
                st.markdown('**Estamos trabalhando com k-nearest Kneighbors. O valor selecionado será proporcional ao número de samples do portifólio!**')
                st.markdown('*Lembrando que quanto maior o valor de K, mais recomendações, porém, menos preciso*')
                slider_nn = st.slider('Número de vizinhos:', 2, 10)

                market_col_select = market_pre[col_select]

                if 'setor' in market_col_select:
                    market_col_select.drop('setor',axis=1,inplace=True)
                    market_col_select=pd.concat([market_col_select,setor_ohe],axis=1)

                if 'nm_meso_regiao' in market_col_select:
                    market_col_select.drop('nm_meso_regiao',axis=1,inplace=True)
                    market_col_select=pd.concat([market_col_select,meso_ohe],axis=1)

                if 'setor' in market_col_select:
                    market_col_select.drop('natureza_juridica_macro',axis=1,inplace=True)
                    market_col_select=pd.concat([market_col_select,juridico_ohe],axis=1)

                market_col_select_scaled=  StandardScaler().fit_transform(market_col_select)
                market_col_select_scaled=pd.DataFrame(market_col_select_scaled,columns=market_col_select.columns,index=market_col_select.index)

                head(market_col_select_scaled)

                st.markdown('## **Recomendação**')
                button_model = st.checkbox('Aperte para iniciar o sistema')

                if button_model:
                    st.text('Loading model...wait!')
                    port_model = market_col_select_scaled.merge(port,how='right',left_index=True,right_index=True)
                    port_model.dropna(inplace=True)
                    suggestion = recommend(port_model,slider_nn,market_col_select_scaled,market)
                    suggestion['id'] = suggestion.index
                    st.text('Loading model...done!')
                    st.markdown('**Sistema de recomendação completo!**')
                    size_sug = suggestion.shape[0]
                    st.write('Foram geraradas ',size_sug,' recomendações!')
                    st.markdown('Baixe aqui:')
                    st.markdown(get_table_download_link(suggestion), unsafe_allow_html=True)
                    coordenadas_market = pd.read_csv('https://raw.githubusercontent.com/guireis1/Codenation-Final-Project/master/data/coordenadas')
                    coordenadas_market.drop('Unnamed: 0',axis=1,inplace=True)
                    cord_reco = geoloc(suggestion,coordenadas_market)
                    cord_reco_df=pd.DataFrame(cord_reco,columns=('lat','lon'))
                    st.markdown('**Geolocalização das empresas recomendadas**')
                    st.map(cord_reco_df)
                    st.markdown('**Visualização das empresas recomendadas**')
                    vis(suggestion)





if __name__ == '__main__':
    main()
