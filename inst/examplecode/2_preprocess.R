# source("inst/examplecode/1_load.R")
d <- read_csv("data-raw/winterActiv.csv") |>
  rename(alpineskiing = `alpine skiing`,
         crosscountryskiing = `cross-country skiing`,
         skitouring = `ski touring`,
         iceskating = `ice-skating`,
         sleighriding = `sleigh riding`,
         horsebackriding = `horseback riding`,
         goingtoaspa = `going to a spa`,
         goingforwalks = `going for walks`,
         organizedexcursions = `organized excursions`,
         goingoutintheevening = `going out in the evening`,
         goingtodiscosbars = `going to discos/bars`,
         sightseeing = `sight-seeing`,
         theateropera = `theater/opera`,
         tyroleanevenings = `tyrolean evenings`,
         localevents = `local events`,
         poolsauna = `pool/sauna`)

d_long <- d |>
  pivot_longer(`alpineskiing`:`poolsauna`,
               names_to = "var",
               values_to = "value")
d_smry <- d_long |>
  group_by(var) |>
  summarise(s = sum(value)) |>
  ungroup() |>
  mutate(p = s/sum(s))

ggplot(d_smry, aes(x=fct_reorder(var, p), y=p)) +
  geom_point() + coord_flip()

keep <- d_smry |>
  filter(p > 0.03) |>
  pull(var)

# Smaller set of variables/activities
d_sub <- d |>
  select(all_of(keep))

# Cluster the data
set.seed(1234) # To match book
clusters = stepcclust(d_sub, k=3, nrep=20)

# Add cluster id to data
d_sub_cl <- d_sub |>
  mutate(cl = as.factor(clusters@second))


