# FeatureCloud App: MI4People_C4R_Embeddings

Diese FeatureCloud App dient der Extraktion des Embessings für die Biological_sample Nodes aus einer biomedizinsichen Neo4j-Wissensdatenbank.  Das Hauptziel ist es, die komplexen, vernetzten Daten für jeden Patienten in einen einzelnen, flachen Vektor umzuwandeln, der sowohl die strukturelle Rolle des Patienten im Graphen als auch seine direkten biologischen Eigenschaften (Proteine, Gene, Varianten etc.) repräsentiert. Es wurde versucht das Skript für dein Einsatz in ressourcesbeschränkten Umgebungen zu optimieren. Das Endprodukt ist eine CSV-Datei, die als direkter Input für nachgelagerte Analysen, wie Patienten-Clustering oder die Identifikation von Biomarkern dienen kann. 

**Wichtiger Hinweis**: Die Entwicklung des Skriptes ist noch nicht abgeschlossen. Ein vollständiger Durchlauf auf dem gesamten Patientendatensatz steht noch aus (aktuell ist das Skript auch bei einer Batch Size von 25 noch gescheitert).

## Vorgehensweise 

Die App folgt einer mehrstufigen Pipeline, um die Daten zu extrahieren und zu verarbeiten: 
1. `Graphen-Projektion`: Zuerst wird mithilfe der GDS Library ein In-Memory-graph erstellt. Der Subgraph enthält alle Patienten (`Biological_sample`) und die direkt mit ihnen verbundenen Entitäten, sowie deren Beziehung untereinander. 
2. `Graphen-Embeddings`: Auf diesem Subgraphen werden zwei Embedding-Algorithmen durchgeführt. 
   - **FastRP**: Erfasst ie globale, strukturelle Rolle jedes Knotens. 
   - **Node2Vec**: Midelliert die lokale Nachbarschaft und den Kontext jedes Knotens. 
Das Ergebnis sind 2 numerische Vektoren pro Patient. Die Embeddings werden als Zwischenergebnis in einer separaten CSV-Datei gespeichert `bio_sample_embeddings_summary.csv`
3. `Metadaten-Extraktion`: Um alle relevanten biologischen Merkmake zu sammlen, wird eine einzelne aber effizeinte und komplexe Cypher-Abfrage in kleinen Batches ausgeführt. Die Abfrage nutzt Subqueries (`Call {}`), um die Daten thematisch zu bündeln und Überlastungen der Datenbank zu vermeiden. 
   - **Protein-Subquery**: Sammelt alle Proteine, die für einen Patienten quantifiziert wurden, sowie deren Verbindung zu molekulaten Funktionen, biologischen Prozessen und Stoffwechselwegen. 
   - **Gen-Subquery**: Sammelt alle Gene, in denen ein Patienten eine Mutation aufweist, und die mit diesen Genen assoziierten Krankheiten. 
   - **Varianten-Subquery**: Findet alle bekannten genetischen Varianten, die an den Proteinen oder Genen eines Patienten gefunden wurden, und prüft, ob diese als klinische relevant makiert sind. 
   - **Patienten-Krankheiten-Subquery**: Sammelt alle Krankheuten, die einem Patienten über die `HAS_DISEASE`-Beziehung zugewiesen sind. 
4. `Finale Zusammenführung`: Die Embeddings werden mit den extrahierten Metadaten zu einer finalen CSV-Datein (`bio_sample_full.csv`) zusammengeführt, in dem jede Zeile einen Patienten repräsentiert. 

# FeatureCloud App Blank Template

The app-blank template contains an initial state that does not execute commands other than transitioning to the terminal state.
This template is a starting point for implementing apps by adding more states and operations.
 

For registering and testing your apps or using other apps, please visit
[FeatureCloud.ai](https://featurecloud.ai/). And for more information about FeatureCloud architecture,
please refer to 
[The FeatureCloud AI Store for Federated Learning in Biomedicine and Beyond](https://arxiv.org/abs/2105.05734) [[1]](#1).


## Developing Apps using FeatureCloud library
FeatureCloud library facilitates app development inside the FeatureCloud platform. To develop apps, developers
should define their states and register them to the default app.

### defining new states
For defining new states, in general, developers can use [`AppState`](engine/README.md#appstate-defining-custom-states)
which supports further communications, transitions, logging, and operations.

#### AppState
[`AppState`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine#appstate-defining-custom-states) is the building block of FeatureCloud apps that covers
all the scenarios with the verifying mechanism. Each state of 
the app should extend [`AppState`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine#appstate-defining-custom-states), which is an abstract class with two specific abstract methods:
- [`register`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#registering-a-specific-transition-for-state-register_transition):
should be implemented by apps to register possible transitions between the current state to other states.
This method is part of verifying mechanism in FeatureCloud apps that ensures logically eligible roles can participate in the current state
and transition to other ones.
- [`run`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#executing-states-computation-run): executes all operations and calls for communication between FeatureCloud clients.
`run` is another part of the verification mechanism in the FeatureCloud library that ensures the transitions to other states are logically correct
by returning the name of the next state.


### Registering apps
For each state, developers should extend one of the abstract states and call the helper function to register automatically
the state in the default FeatureCloud app:

```angular2html
@app_state(name='initial', role=Role.BOTH, app_name='example')
class ExampleState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        self.read_config()
        self.app.log(self.config)
        return 'terminal'
```

### building the app docker image
Once app implementation is done, building the docker image for testing or adding it to
[FeatureCloud AI store](https://featurecloud.ai/ai-store?view=store&q=&r=0),
developers should provide the following files.
#### Dockerization files

For dockerizing apps, regardless of their applications, there should be some specific files:

1. [Dockerfile](Dockerfile)
2. [server-config](server_config)
   - [docker-entrypoint.sh](server_config/docker-entrypoint.sh)
   - [nginx](server_config/nginx)
   - [supervisord.conf](server_config/supervisord.conf)

Developers should ensure that these files with the same structure and content exist in the same directory as their app
implementation. 


#### App-specific files
All app-specific files should include data or codes strictly dependent on the app's functionality.

##### main.py
Each app should be implemented in a directory that includes the [`main.py`](main.py) file, which in turn comprises either direct
implementation of states or importing them. Moreover, `main` should import `bottle` and `api` packages:
```angular2html
from bottle import Bottle

from api.http_ctrl import api_server
from api.http_web import web_server

import apps.examples.dice

from engine.app import app

server = Bottle()
```
One can implement desired states in [`states.py`](states.py) and import it, which because of putting 
[`app_state`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#registering-states-to-the-app-app_state) on top of state classes, 
merely importing the states and registering them into the [`app` instance](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#app-instance).     

For running the app, inside a docker container, [`app.register()`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#registering-all-transitions-appregister)
should be called to register and verify all transitions; next, api and servers should mount at corresponding paths; and finally
the server is ready to run the app.

```angular2html
    app.register()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host='localhost', port=5000)
```

All of the codes above, except for importing the app or, alternatively, implementing states, can be exactly same for all apps.  

##### requirements.txt
for installing required python libraries inside the docker image, developers should provide a list of libraries in [requirements.txt](requirements.txt).
Some requirements are necessary for the FeatureCloud library, which should always be listed, are:
```angular2html
bottle
jsonpickle
joblib
numpy
bios
pydot
pyyaml
```

And the rest should be all other app-required libraries.

##### config.yml
Each app may need some hyper-parameters or arguments that the end-users should provide. Such data should be included
in [`config.yml`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app#config-file-configyml), which should be read and interpreted by the app. 

### Run YOUR_APPLICATION

#### Prerequisite

To run YOUR_APPLICATION, you should install Docker and FeatureCloud pip package:

```shell
pip install featurecloud
```

Then either download YOUR_APPLICATION image from the FeatureCloud docker repository:

```shell
featurecloud app download featurecloud.ai/YOUR_APPLICATION
```

Or build the app locally:

```shell
featurecloud app build featurecloud.ai/YOUR_APPLICATION
```

Please provide example data so others can run YOUR_APPLICATION with the desired settings in the `config.yml` file.

#### Run YOUR_APPLICATION in the test-bed

You can run YOUR_APPLICATION as a standalone app in the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also run the app using CLI:

```shell
featurecloud test start --app-image featurecloud.ai/YOUR_APPLICATION --client-dirs './sample/c1,./sample/c2' --generic-dir './sample/generic'
```



### References
<a id="1">[1]</a> 
Matschinske, J., Späth, J., Nasirigerdeh, R., Torkzadehmahani, R., Hartebrodt, A., Orbán, B., Fejér, S., Zolotareva,
O., Bakhtiari, M., Bihari, B. and Bloice, M., 2021.
The FeatureCloud AI Store for Federated Learning in Biomedicine and Beyond. arXiv preprint arXiv:2105.05734.
