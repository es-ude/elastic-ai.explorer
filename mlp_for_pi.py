from elasticai.explorer import explorer
from elasticai.generator.generator import PIGenerator

if __name__ == '__main__':
    top_models=explorer.search()
    generator= PIGenerator()
    for i, top_model in enumerate(top_models):
        generator.generate(top_model, path="models/ts_models/model " +str(i))
