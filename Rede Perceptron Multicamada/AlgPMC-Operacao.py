from PMC import PMC
import pandas as pd
import numpy as np
import csv


with open("Rede Perceptron Multicamada/dados-validacao.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    dados = list(reader)


for cj_treinamento in range(1, 6):

    #escolhe o conjunto de treinamento
    PESOS = cj_treinamento

    resultados = []


    for amostra_dado in dados:
        # x1, x2, x3 vindos do dados-validacao.csv
        X = [float(x) for x in amostra_dado]


        W_T1 = [{'id': 'L1N1', 'weights': [0.9239667567544722, 1.6834689044558635, 0.7160805371133888], 'bias': 2.3388710104143233}, {'id': 'L1N2', 'weights': [1.2094123361481588, 0.9272817155234028, 1.1119716735284428], 'bias': 1.9372847143798746}, {'id': 'L1N3', 'weights': [1.3208599165565638, 0.8785399890465284, 1.2128562193335028], 'bias': 2.379487406153142}, {'id': 'L1N4', 'weights': [1.4976215331745364, 1.0159394436013471, 0.9657774989169498], 'bias': 2.2608576198356904}, {'id': 'L1N5', 'weights': [1.8648235459036482, 1.5905342212225464, 1.4397497517413882], 'bias': 2.4549091054027694}, {'id': 'L1N6', 'weights': [1.0887639459766134, 1.5265216955110792, 0.9311030653449278], 'bias': 2.477940709139768}, {'id': 'L1N7', 'weights': [1.1731671415615625, 1.0214829277681625, 1.64544564899317], 'bias': 2.362803651897099}, {'id': 'L1N8', 'weights': [1.1753724653901767, 1.2547246530419, 1.1813864304671395], 'bias': 1.7211237922588858}, {'id': 'L1N9', 'weights': [1.1312506504270703, 1.644195568213406, 1.634718397905425], 'bias': 2.0112878111278927}, {'id': 'L1N10', 'weights': [1.582419662922055, 1.3224148190900689, 1.3303447226710383], 'bias': 1.8036237902305092}, {'id': 'L2N1', 'weights': [0.7258296633159868, 0.13595294617404902, 0.5400338876766364, 0.519787104048029, 0.6703570930780215, 0.31199614537821435, 0.9240732459734264, 0.06243598815127965, 0.7700517892332239, 0.8249470319518586], 'bias': 1.8308005057314187}]


        W_T2 = [{'id': 'L1N1', 'weights': [1.0537964762830745, 1.2859077198099087, 1.2351289098381273], 'bias': 2.736896039683494}, {'id': 'L1N2', 'weights': [1.2834768579748057, 1.635429314101757, 1.1200238392273165], 'bias': 1.908482412024482}, {'id': 'L1N3', 'weights': [0.9989176305732533, 0.8393268478928919, 1.32625724072169], 'bias': 2.589518583240802}, {'id': 'L1N4', 'weights': [1.2187458923317993, 1.2478100614853889, 0.9832974048054222], 'bias': 2.46404838981803}, {'id': 'L1N5', 'weights': [1.616011248291235, 0.8214959999216869, 1.325740170501031], 'bias': 2.016087448683278}, {'id': 'L1N6', 'weights': [0.6727216128102539, 1.0529700594934444, 1.4128334132079272], 'bias': 2.0368399006623554}, {'id': 'L1N7', 'weights': [1.482211790072221, 1.2871328928422374, 1.14468371710358], 'bias': 2.4652556916357997}, {'id': 'L1N8', 'weights': [0.7960467507856394, 1.610427607067242, 0.8512578290360278], 'bias': 2.7203666511805564}, {'id': 'L1N9', 'weights': [1.0862850823772958, 1.2367254036937803, 0.8091568305059011], 'bias': 2.485600792052701}, {'id': 'L1N10', 'weights': [1.5894119812503704, 1.1109684648973428, 1.270047462365177], 'bias': 2.322834225332303}, {'id': 'L2N1', 'weights': [0.6126985616410312, 0.5261445243238662, 0.6076719151284288, 0.5790145620978372, 0.4148114523325231, 0.7786966465287914, 1.0223575936198506, 0.6425755351832844, 0.35060562581306415, 0.8814018897599015], 'bias': 1.7568238620264631}]

        W_T3 = [{'id': 'L1N1', 'weights': [0.9338579591775348, 0.8574837825746967, 1.3388986721902916], 'bias': 1.68990499519814}, {'id': 'L1N2', 'weights': [0.9216743888728925, 1.5734015809735435, 1.288826478592611], 'bias': 2.453453973237017}, {'id': 'L1N3', 'weights': [1.0528307033897586, 0.9087411900368317, 1.5457318073712465], 'bias': 2.411648991488267}, {'id': 'L1N4', 'weights': [0.8934172450904524, 1.505804172020584, 0.7003489830153135], 'bias': 2.0245334061081937}, {'id': 'L1N5', 'weights': [1.5257954159571723, 1.5585292027084492, 1.026601423111165], 'bias': 2.3741529447968954}, {'id': 'L1N6', 'weights': [1.1186103203172335, 1.087114960335224, 1.441538840060131], 'bias': 1.8800110316746739}, {'id': 'L1N7', 'weights': [0.8245889071268692, 1.2247974409342888, 0.8213285476267855], 'bias': 1.7433771108623797}, {'id': 'L1N8', 'weights': [1.5290456793897182, 1.2651867721004058, 0.9679748516501502], 'bias': 1.7573162422120714}, {'id': 'L1N9', 'weights': [1.42740879791037, 1.269999098652585, 0.772667237932054], 'bias': 2.2936601188369483}, {'id': 'L1N10', 'weights': [1.5399197654555015, 1.3272402324533097, 1.5834389688709847], 'bias': 2.3660309669590287}, {'id': 'L2N1', 'weights': [0.3927641997014695, 0.7142163337005621, 0.44327752521559843, 0.16890800250037571, 0.4465195452718836, 0.6944804903340852, 0.2504237459214505, 0.4897820685733438, 0.8967809859420687, 1.1779484383587637], 'bias': 1.85738129253734}] 

        W_T4 = [{'id': 'L1N1', 'weights': [1.656203882593353, 1.2326937193427059, 1.5008133343303116], 'bias': 1.6460124559953149}, {'id': 'L1N2', 'weights': [1.4615793864068538, 1.1295450138386554, 1.5078186056884306], 'bias': 2.430201215688516}, {'id': 'L1N3', 'weights': [1.4443064206607177, 1.7158835473909972, 0.7678200769648647], 'bias': 1.7064048894935644}, {'id': 'L1N4', 'weights': [1.2430237572663787, 1.238319125847032, 1.4614093174294107], 'bias': 2.2879616902112287}, {'id': 'L1N5', 'weights': [1.532548628920244, 1.717843580311778, 1.2444724192956589], 'bias': 2.5295504901588837}, {'id': 'L1N6', 'weights': [1.2514667530661325, 1.0422079317180066, 0.8451230503853975], 'bias': 2.2118085587296568}, {'id': 'L1N7', 'weights': [0.875898577169137, 1.1248897869382302, 1.190796638051762], 'bias': 2.5286130173056813}, {'id': 'L1N8', 'weights': [1.260185950887992, 1.4267244484498838, 0.9297230686344748], 'bias': 2.436570192876373}, {'id': 'L1N9', 'weights': [1.3269612838710219, 1.2316517749178066, 1.2227987676276475], 'bias': 1.7867774789969173}, {'id': 'L1N10', 'weights': [0.9031848393645414, 1.1987071180327211, 1.4809845064201712], 'bias': 2.1501209727516564}, {'id': 'L2N1', 'weights': [0.2931243440141543, 0.908265121419675, 0.07920288572187148, 0.5059149626247943, 0.9270825314956045, 0.4276911767631267, 0.1369984944904732, 0.8131816568606285, 0.6738751157468165, 0.7924313909183379], 'bias': 1.792853796794459}]

        W_T5 = [{'id': 'L1N1', 'weights': [0.9059169046750186, 1.0568773470348174, 0.800976912424847], 'bias': 1.7202455909558072}, {'id': 'L1N2', 'weights': [1.487103605385983, 1.1167799024079983, 1.2530680635702014], 'bias': 2.3859718702263035}, {'id': 'L1N3', 'weights': [0.9673000327812913, 1.2034308016167956, 0.6414620012957716], 'bias': 1.7472106150766495}, {'id': 'L1N4', 'weights': [0.965333170329354, 1.3441203655838745, 1.5938530430890618], 'bias': 1.6647242718555137}, {'id': 'L1N5', 'weights': [1.470558007463779, 1.1743963767747658, 1.5958045204612756], 'bias': 1.6306347422332883}, {'id': 'L1N6', 'weights': [1.4135340523437585, 1.2930808900146817, 1.5917823139746963], 'bias': 1.6235239808300528}, {'id': 'L1N7', 'weights': [1.1232342379458862, 0.9620597584960222, 1.2913817597493304], 'bias': 1.8833287923795452}, {'id': 'L1N8', 'weights': [0.9438214322295224, 1.0789133482630096, 0.8127157544279225], 'bias': 1.6717627661736874}, {'id': 'L1N9', 'weights': [1.592535105906074, 1.517095318839147, 1.5225767666436645], 'bias': 2.3664513305063823}, {'id': 'L1N10', 'weights': [1.1376929391730752, 1.8254193223749282, 0.8497169182720247], 'bias': 1.6714962279901777}, {'id': 'L2N1', 'weights': [0.7419832488835509, 0.6622760158978162, 0.5476134923584915, 0.17540514541541552, 0.9757669128748127, 0.6800901617616654, 0.5292584233703417, 0.5300966970397516, 0.4411288417009682, 0.420789543017626], 'bias': 2.2438879074961524}]
        W_TX = [W_T1, W_T2, W_T3, W_T4, W_T5]


        STRUCTURE = [3, 10, 1]

        # Instanciar o perceptron
        pmc = PMC(STRUCTURE)  # STRUCTURE = [3, 10, 1] 3 - inputs ; 10 - hidden ; 1 - output

        pmc.import_weights_and_bias(W_TX[cj_treinamento-1])

        y = pmc.predict(X)

        resultados.append(y[0])

    print(f"Resultados T{cj_treinamento}: {resultados}\n\n")
          

