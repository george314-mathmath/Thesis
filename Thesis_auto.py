
# Set up
import numpy as np
import itertools  
import random
import os

from pylatex import (
    Alignat,
    Axis,
    Document,
    Figure,
    Math,
    Matrix,
    Plot,
    Section,
    Subsection,
    Tabular,
    TikZ,
)
from pylatex.utils import italic

# End of Set up









def generate_s_intervals(M,N):
    #outputs a dictionary 'support'
    support_M = {}
    support_N = {}
    temp={}
    i=0
    j=0
    for x in M:
        while j < len(M):
            output = find_s(x,M[j])
            temp[j]=output
            support_M[i] = temp
            j+=1
        j=0

        while j < len(N):
            output = find_s(x,N[j])
            temp[j]=output
            support_M[i] = temp
            j+=1
        j=0
        i+=1

    i=0
    j=0
    for a in N:
        for b in M:
            output = find_s(a,b)
            temp[j]=output
            support_N[i] = temp
            j+=1
        j=0
        for c in N:
            output = find_s(a,c)
            temp[j]=output
            support_N[i] = temp
            j+=1
        j=0
        i+=1
    return support_M,support_N

def find_s(i,j,precision=6):
    sup = [0,1]
    if (j[0] <=i[0]):
        if (j[1] <=i[0]):
            sup = False
            return sup
        elif (j[1] <=i[1]):
            sup[0]= 0 
            sup[1]=round(j[1]-i[0],precision)
            return sup
        else:
            sup[0]= round(j[1]-i[1],precision)
            sup[1]= round(i[1]-i[0],precision)+round(j[1]-i[1],precision)
            return sup
    elif (j[0] <=i[1]):
        if (j[1] <=i[1]):
            sup[0]= round(j[0]-i[0],precision)
            sup[1]= round(j[1]-i[0],precision)
            return sup
        else:
            sup[0]=round(max(j[0]-i[0],j[1]-i[1]),precision)
            sup[1]=round(j[1]-i[0],precision)
            return sup
    else:
        sup[0]=round(max(j[0]-i[0],j[1]-i[1]),precision)
        sup[1]=round(j[1]-i[0],precision)
        return sup


def print_intervals(G,P,precision=2):
    for x in G:
        j=0
        while j < len(G):
            output = find_s(x,G[j],precision)
            print("M:",x," ","M:", G[j]," S:" ,output) 
            j+=1
        j=0

        while j < len(P):
            output = find_s(x,P[j],precision)
            print("M:",x," ","N:", P[j]," S:" ,output)  
            j+=1
        j=0

    for x in P:
        j=0
        while j < len(G):
            output = find_s(x,G[j],precision)
            print("N:",x," ","M:", G[j]," S:" ,output) 
            j+=1
        j=0

        while j < len(P):
            output = find_s(x,P[j],precision)
            print("N:",x," ","N:", P[j]," S:" ,output)  
            j+=1
        j=0


    return


    
def get_epsilon(M,N):
    epsilons = []
    epsilons_unique=[]
    i=0
    j=0
    for x in M:
        while j < len(M):
            output = find_s(x,M[j])
            if output != False:
                #epsilons.append(output[0])
                #epsilons.append(output[1])
                epsilons.append(output[0]/2)
                epsilons.append(output[1]/2)
            j+=1
        j=0

        while j < len(N):
            output = find_s(x,N[j])
            if output != False:
                epsilons.append(output[0])
                epsilons.append(output[1])
                #epsilons.append(output[0]/2)
                #epsilons.append(output[1]/2)
            j+=1
        j=0
        i+=1

    i=0
    j=0
    for a in N:
        for b in M:
            output = find_s(a,b)
            if output != False:
                epsilons.append(output[0])
                epsilons.append(output[1])
                #epsilons.append(output[0]/2)
                #epsilons.append(output[1]/2)
            j+=1
        j=0
        for c in N:
            output = find_s(a,c)
            if output != False:
                #epsilons.append(output[0])
                #epsilons.append(output[1])
                epsilons.append(output[0]/2)
                epsilons.append(output[1]/2)
            j+=1
        j=0
        i+=1
    for a in epsilons:
        if a not in epsilons_unique:
            epsilons_unique.append(a)
    epsilons_unique.sort()
    return epsilons_unique

def matricies(M,N,epsilon,precision=2):
    m_to_n = []
    n_to_m = []
    m_to_2epsilon_m = []
    n_to_2epsilon_n = []
    m_i_to_n_i =[]
    n_i_to_m_i =[]
    m_i_to_m_i =[]
    n_i_to_n_i =[]
    two_epsilion = epsilon*2

    for i in range(len(M)):
            m_i_to_n_i.append("_")
            m_i_to_m_i.append("_")
    for ht in range(len(N)):
            n_i_to_m_i.append("_")
            n_i_to_n_i.append("_")
    tmp = m_i_to_n_i.copy()
    for j in range(len(N)):
        m_to_n.append(tmp.copy())
        n_to_2epsilon_n.append(tmp.copy())

    tmp = n_i_to_m_i.copy()
    for t in range(len(M)):
        n_to_m.append(tmp.copy())
        m_to_2epsilon_m.append(tmp.copy())



    which_m = 0
    for x in M:
        #print(which_m)
        j=0
        while j < len(M):
            output = find_s(x,M[j],precision)
            if output == False:
                m_to_2epsilon_m[j][which_m]= "_"
            elif (two_epsilion >= output[0])&(two_epsilion <= output[1]):
                if which_m == j:
                    m_to_2epsilon_m[j][which_m]= "1"
                else:
                    m_to_2epsilon_m[j][which_m]= "0"
            j+=1
        j=0

        while j < len(N):
            output = find_s(x,N[j],precision)
            if output == False:
                m_to_n[j][which_m]= "0"
            elif (epsilon >= output[0])&(epsilon <= output[1]):
                m_to_n[j][which_m]= "k"
            else:
                m_to_n[j][which_m]= "0"  
            j+=1
        
        j=0
        which_m =which_m+1

    which_n = 0
    for x in N:
        j=0
        while j < len(M):
            output = find_s(x,M[j],precision)
            if output == False:
                n_to_m[j][which_n]= "0"
            elif (epsilon >= output[0])&(epsilon <= output[1]):
                n_to_m[j][which_n]= "l"
            else:
                n_to_m[j][which_n]= "0" 
            j+= 1
        j=0

        while j < len(N):
            output = find_s(x,N[j],precision)
            if output == False:
                n_to_2epsilon_n[j][which_n]= "_"
            elif (two_epsilion >= output[0])&(two_epsilion <= output[1]):
                if which_n == j:
                    n_to_2epsilon_n[j][which_n]= "1"
                else:
                    n_to_2epsilon_n[j][which_n]= "0" 

            j+=1
        j=0
        which_n +=1

    print("The matrix from M to N:")
    for v in range(len(N)):
        print(m_to_n[v])
    print("The matrix from N to M:")
    for i in range(len(M)):
        print(n_to_m[i])
    print("The matrix from M to 2 epsilon M:")
    for i in range(len(M)):
        print(m_to_2epsilon_m[i])
    print("The matrix from N to 2 epsilon N:")
    for i in range(len(N)):
        print(n_to_2epsilon_n[i])
    return 



def print_major(M,N):
    epsilons_to_itterate = get_epsilon(M,N)
    for i in range(len(epsilons_to_itterate)):
        print("Epsilon is ", epsilons_to_itterate[i])
        matricies(M,N, epsilons_to_itterate[i])
        print("")
        print("")
        print("")


def interleaving(M,N):
    #First find the width of each interval
    cost_nomatch_m = []
    cost_nomatch_n = []
    for i in range(len(M)):
        cost_nomatch_m.append(round((M[i][1]-M[i][0])/2, 4))
    for i in range(len(N)):
        cost_nomatch_n.append(round((N[i][1]-N[i][0])/2,4))
    # We have the cost of not matching now
    #Create temp arrays to store the indexs 
    m_index = []
    n_index = []
    for i in range(len(M)):
        m_index.append(i)
    for g in range(len(N)):
        n_index.append(g)

    # Lets create a variable store optimal match and optimal match cost
    optimal_match = []
    counter=0
    optimal_match_cost = max(max(cost_nomatch_m),max(cost_nomatch_n))
    # Lets create a temp variable to store current match and current match cost
    # potential match is array of arrays of the form [i,j] is the ith elemnet of m 
    # matched with the jth element.
    
    potential_match_total_cost = 0
    match = [0,0]
    for i in range(1,min(len(M),len(N))+1):
        for subset_m in itertools.combinations(m_index, i):
            for subset_n in itertools.permutations(n_index, i):
                potential_match = []
                potential_match_costs = []
                m_index_matched =[]
                n_index_matched=[]
                for g in range(i):
                    match[0]= subset_m[g]
                    match[1]= subset_n[g]
                    potential_match.append(match.copy())
                    m_index_matched.append(subset_m[g])
                    n_index_matched.append(subset_n[g])
                for k in range(i):
                    potential_match_costs.append(cost(potential_match[k][0],potential_match[k][1],M,N))
                potential_match_total_cost = max(unmatchcost(m_index_matched,cost_nomatch_m),unmatchcost(n_index_matched,cost_nomatch_n),max(potential_match_costs))
                if potential_match_total_cost<optimal_match_cost:
                    optimal_match = potential_match
                    optimal_match_cost = potential_match_total_cost
                #If optimal cost less than unmatched cost break out of loop
                #Have a version which does not
                counter+=1
                #print("Macthing cost:",potential_match_total_cost)
    #print("counter:",counter)
    return optimal_match,round(optimal_match_cost,3)
    
def cost(p,k,M,N):
    return max(abs(M[p][0]-N[k][0]),abs(M[p][1]-N[k][1]))

def unmatchcost(index,M):
    temp = [0]
    for i in range(len(M)):
        if i not in index:
            temp.append(M[i])
    return max(temp)


def interleaving_print(M,N):
    matches,distnace = interleaving(M,N)
    print("The interleaving distnace is: ", distnace)
    for match in matches:
        print("Element ",M[match[0]], 
              "of M matches with element ",
              N[match[1]]," of N" )


def double_indexing(M,N):
    for x in M:
        j=0
        while j < len(M):
            output = find_s(x,M[j])
            print("M:",x," ","M:", M[j]," S:" ,output) 
            j+=1
        j=0

        while j < len(N):
            output = find_s(x,N[j])
            print("M:",x," ","N:", N[j]," S:" ,output)  
            j+=1
        j=0

    for x in N:
        j=0
        while j < len(M):
            output = find_s(x,M[j])
            print("N:",x," ","M:", M[j]," S:" ,output) 
            j+=1
        j=0

        while j < len(N):
            output = find_s(x,N[j])
            print("N:",x," ","N:", N[j]," S:" ,output)  
            j+=1
        j=0
    return


def matricies_no_print(M,N,epsilon,precision=6):
    m_to_n = []
    n_to_m = []
    m_to_2epsilon_m = []
    n_to_2epsilon_n = []
    m_i_to_n_i =[]
    n_i_to_m_i =[]
    m_i_to_m_i =[]
    n_i_to_n_i =[]
    two_epsilion = epsilon*2

    for i in range(len(M)):
            m_i_to_n_i.append("-")
            m_i_to_m_i.append("-")
    for ht in range(len(N)):
            n_i_to_m_i.append("-")
            n_i_to_n_i.append("-")

    for j in range(len(N)):
        tmp = m_i_to_n_i.copy()
        tmp1 =n_i_to_n_i.copy()
        m_to_n.append(tmp.copy())
        n_to_2epsilon_n.append(tmp1.copy())

    for t in range(len(M)):
        tmp = n_i_to_m_i.copy()
        tmp1 = m_i_to_m_i.copy()
        n_to_m.append(tmp.copy())
        m_to_2epsilon_m.append(tmp1.copy())

     

    which_m = 0
    for x in M:
        #print(which_m)
        j=0
        while j < len(M):
            output = find_s(x,M[j],precision)
            if output == False:
                m_to_2epsilon_m[j][which_m]= "-"
            elif (two_epsilion+0.0000000001 >= output[0])&(two_epsilion+0.0000000001 <= output[1]):
                if which_m == j:
                    m_to_2epsilon_m[j][which_m]= "1"
                else:
                    m_to_2epsilon_m[j][which_m]= "0"
            j+=1
        j=0

        while j < len(N):
            output = find_s(x,N[j],precision)
            if output == False:
                m_to_n[j][which_m]= "0"
            elif (epsilon+0.0000000001 >= output[0])&(epsilon+0.0000000001 <= output[1]):
                m_to_n[j][which_m]= "k"
            else:
                m_to_n[j][which_m]= "0"  
            j+=1
        
        j=0
        which_m =which_m+1

    which_n = 0
    for x in N:
        j=0
        while j < len(M):
            output = find_s(x,M[j],precision)
            if output == False:
                n_to_m[j][which_n]= "0"
            elif (epsilon+0.0000000001 >= output[0])&(epsilon+0.0000000001 <= output[1]):
                n_to_m[j][which_n]= "l"
            else:
                n_to_m[j][which_n]= "0" 
            j+= 1
        j=0

        while j < len(N):
            output = find_s(x,N[j],precision)
            if output == False:
                n_to_2epsilon_n[j][which_n]= "-"
            elif (two_epsilion+0.0000000001 >= output[0])&(two_epsilion+0.0000000001 <= output[1]):
                if which_n == j:
                    n_to_2epsilon_n[j][which_n]= "1"
                else:
                    n_to_2epsilon_n[j][which_n]= "0" 
            j+=1
        j=0
        which_n +=1

    #print("The matrix from M to N:")
    #for v in range(len(N)):
        #print(m_to_n[v])
    #print("The matrix from N to M:")
    #for i in range(len(M)):
        #print(n_to_m[i])
    #print("The matrix from M to 2 epsilon M:")
    #for i in range(len(M)):
        #print(m_to_2epsilon_m[i])
    #print("The matrix from N to 2 epsilon N:")
    #for i in range(len(N)):
        #print(n_to_2epsilon_n[i])
    return m_to_n,n_to_m,m_to_2epsilon_m,n_to_2epsilon_n


def generate_mn(left_endpoint_min=0,left_endpoint_max=2,width_max=2,m_size=4,n_size=4):
    x=[]
    y=[]
    for i in range(m_size):
        left_point = random.uniform(left_endpoint_min, left_endpoint_max)
        right_point = left_point+random.uniform(0,width_max)
        x+=[[round(left_point,3),round(right_point,3)]]
    for j in range(n_size):
        left_point = random.uniform(left_endpoint_min, left_endpoint_max)
        right_point = left_point+random.uniform(0,width_max)
        y+=[[round(left_point,3),round(right_point,3)]]
    return x,y



def single_equations(m,k_or_l):
    i=0
    answer = []
    while i < len(m):
        j=0
        while j < len(m[i]):
            if m[i][j] == "0":
                thing_to_add = str(k_or_l)+str(j+1)+str(i+1)+"=0"
                answer.append(thing_to_add)
            j+=1
        i+=1
    return answer

def double_equations(m_to_n,n_to_m,m_to_2m,n_to_2n):
    i=0
    answer_m = []
    answer_n = []
    temp_m = np.array(m_to_n)
    temp_m = np.transpose(temp_m)
    temp_n = np.array(n_to_m)
    temp_n = np.transpose(temp_n)

    while i < len(m_to_2m):
        j=0
        while j < len(m_to_2m[i]):
            if (m_to_2m[i][j] == "0"):
                thing_to_add = add_strs_row_col(n_to_m[i],temp_m[j],i,j)
                if thing_to_add != "":
                    answer_m.append(thing_to_add)
            if (m_to_2m[i][j] == "1"):
                thing_to_add = add_strs_row_col(n_to_m[i],temp_m[j],i,j)
                if thing_to_add != "":
                    thing_to_add += "-1"
                answer_m.append(thing_to_add)
            j+=1
        i+=1
    
    i=0
    while i < len(n_to_2n):
        j=0
        while j < len(n_to_2n[i]):
            if (n_to_2n[i][j] == "0"):
                thing_to_add = add_strs_row_col(m_to_n[i],temp_n[j],i,j)
                if thing_to_add != "":
                    answer_n.append(thing_to_add)
            if (n_to_2n[i][j] == "1"):
                thing_to_add = add_strs_row_col(m_to_n[i],temp_n[j],i,j)
                if thing_to_add != "":
                    thing_to_add += "-1"
                answer_n.append(thing_to_add)
            j+=1
        i+=1
    return answer_m, answer_n

def add_strs_row_col(x,y,i,j):
    string_base=""
    g=0
    while g < len(x):
        if (x[g] != "0"):
            if (y[g] != "0"):
                string_base += str(x[g])+str(g+1)+str(i+1)+"*"+y[g]+str(j+1)+str(g+1)
                if g != (len(x)-1):
                    string_base+="+"
        g+=1
    if (string_base != ""):
        if (string_base[-1] == "+"):
            string_base = string_base[:-1]
    return string_base



def generate_tex_documents(left_endpoint_min,left_endpoint_max,width_max,m_size,n_size,how_many_times,name):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    for i in range(0,how_many_times):
        x,y =generate_mn(left_endpoint_min,left_endpoint_max,width_max,m_size,n_size)
        match,dist = interleaving(x,y)
        doc = Document(default_filepath =  "Put your desired filepath here",geometry_options=geometry_options)
        with doc.create(Section("The overall info")):
            doc.append("Here we have M is")
            doc.append(Math(data=[np.array(x)]))
            doc.append("\n And we have N is")
            doc.append(Math(data=[np.array(y)]))
            epsilons_to_itterate = get_epsilon(x,y)
            doc.append("\n And we have the relevant epsilon are")
            doc.append(Math(data=[np.array(epsilons_to_itterate)]))
            doc.append("\n And we have the interleaving distance is")
            doc.append(Math(data=[dist]))
            doc.append("\n And we have the matching is")
            doc.append(Math(data=[np.array(match)]))
        with doc.create(Section("The progression of the matricies")):
            for j in range(len(epsilons_to_itterate)):
                a,b,c,d = matricies_no_print(x,y, epsilons_to_itterate[j])
                den,dem = double_equations(a,b,c,d)
                sem,sen = single_equations(a,"k"),single_equations(b,"l")
                a = np.matrix(a) #m_to_n
                b = np.matrix(b) #n_to_m
                c = np.matrix(c) #m_to_2epsilon_m
                d = np.matrix(d) #n_to_2epsilon_n
                epsilon_current = str(epsilons_to_itterate[j])
                with doc.create(Subsection("For epsilon equal to "+epsilon_current)):
                    doc.append("For N to 2 epsilon N")
                    doc.append(Math(data=[Matrix(a), Matrix(b), "=", Matrix(d)]))
                    doc.append("\nFor M to 2 epsilon N")
                    doc.append(Math(data=[Matrix(b), Matrix(a), "=", Matrix(c)]))
                    doc.append("\nThe variety is:")
                    doc.append("\nSingle value requirements for M to N matrix:\n")
                    for j in range(0,len(sem)):
                        doc.append(sem[j]+"\n")
                    doc.append("\nSingle value requirements for M to N matrix:\n")
                    for j in range(0,len(sen)):
                        doc.append(sen[j]+"\n")
                    doc.append("\nMulti value requirements for M to 2M:\n")
                    for j in range(0,len(dem)):
                        doc.append(dem[j]+"\n")
                    doc.append("\nMulti value requirements for N to 2N:\n")
                    for j in range(0,len(den)):
                        doc.append(den[j]+"\n")
                    #doc.append("\n")
                    #doc.append("\n")
                    #doc.append("\n")
        temp_name = str(name)+"_"+str(i)
        doc.generate_pdf(temp_name, clean_tex=True)
    return
